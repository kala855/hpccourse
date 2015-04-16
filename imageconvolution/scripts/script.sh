#!/bin/bash

for i in {1..9}
do
    for j in {1..100}
    do
        ./opencvgpu ../../images/img$i.jpg >> test$i.csv
    done
    echo "Ready for image img$i.jpg"
done
