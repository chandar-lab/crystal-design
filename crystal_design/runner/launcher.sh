#!/bin/bash
​
for weight in 5
do
    for beta in 1 3
    do
        for alpha2 in 5 10
        do
            for alpha1 in 1
            do
                name="112megnet-MG-w"$weight-$alpha1-$alpha2-$beta
                sbatch script $alpha1 $alpha2 $beta $weight $name
            done
        done
    done
done
