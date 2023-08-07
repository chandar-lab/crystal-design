#!/bin/bash
for i in {1362..27135}; 
    do 
        mgl predict -m MEGNet-MP-2019.4.1-BandGap-mfi --infile mp_20_cifs/$i.cif > matgl_pred_bg_mp_train/$i.txt
    done