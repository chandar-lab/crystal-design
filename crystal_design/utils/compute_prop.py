from collections import Counter
import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
# from p_tqdm import p_map
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
import sys
# sys.path.append('/home/mila/p/prashant.govindarajan/scratch/COMP760-Project/cdvae')
sys.path.append('/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/cdvae')

from scripts.eval_utils import (smact_validity, structure_validity, CompScaler, get_fp_pdist,
                                    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords.cpu(), coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class OptEval(object):

    def __init__(self, crys, num_opt=1, eval_model_name='perovskite'):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])#[:self.step_opt * self.num_opt]
        valid_crys = [c for c in self.crys if c.valid]#[:valid_x.shape[0]]
        # valid_crys[0].dict['frac_coords'] = valid_crys[0].dict['frac_coords'].cpu()
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            pred_props = prop_model_eval(self.eval_model_name, [c.dict for c in valid_crys])
            return pred_props

    def get_metrics(self):
        props = self.get_success_rate()
        percentiles = np.array([0.43924842, 0.61202443, 0.7364607])
        props_array = np.array(props)
        sr_5 = (props_array <= percentiles[0]).mean()
        sr_10 = (props_array <= percentiles[1]).mean()
        sr_15 = (props_array <= percentiles[2]).mean()
        return props, (sr_5, sr_10, sr_15)

class GenEval(object):

    def __init__(self, pred_crys, eval_model_name='perovskite'): ##CHANGE N_SAMPLES to 5
        self.crys = pred_crys
        # self.gt_crys = gt_crys
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        n_samples = len(valid_crys) #n_samples 
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        # metrics.update(self.get_comp_diversity())
        # metrics.update(self.get_struct_diversity())
        # metrics.update(self.get_density_wdist())
        # metrics.update(self.get_num_elem_wdist())
        # metrics.update(self.get_prop_wdist())
        # print(metrics)
        # metrics.update(self.get_coverage())
        return metrics

