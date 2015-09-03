#!/bin/bash

WEKA_PATH=.

CP="$CLASSPATH:$WEKA_PATH/weka.jar"

 java -cp $CP  weka.filters.unsupervised.attribute.Remove -R $3 -i $1 -o $2
