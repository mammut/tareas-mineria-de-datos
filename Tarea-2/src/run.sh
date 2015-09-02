#!/bin/bash

WEKA_PATH=.

CP="$CLASSPATH:$WEKA_PATH/weka.jar"
if [[ "$5" -eq 1  ]]; then
    echo "Apriori"
    java -cp $CP weka.associations.Apriori -N 5000 -T 0 -C $3 -D 0.05 -U 1.0 -M $4 -S -1.0 -c -1 -t $1 > $2
else
    echo "FP-growth"
    java -cp $CP weka.associations.FPGrowth -P 2 -I -1 -N 5000 -T 0 -C $3 -D 0.05 -U 1.0 -M $4 -t $1 > $2
fi
