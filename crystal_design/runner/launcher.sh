#!/bin/bash
​
si_bg=1.12
for weight in 1 5
do
    for beta in 1 3
    do
        for alpha2 in 5 10
        do
            for alpha1 in 1 0
            do
                name="112megnet-MG-w"$weight-$alpha1-$alpha2-$beta ## IF BG = 1.12
                # name="megnet-MG-w"$weight-\($alpha1-$alpha2-$beta\)"-bg4-nq1"   ## IF BG = 4.0
                sbatch script $alpha1 $alpha2 $beta $weight $name $si_bg
            done
        done
    done
done
