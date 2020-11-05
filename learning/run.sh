#!/bin/bash

# cd ../src
# make
# make preprocess
# cd ../learning

# python make_data.py $1
# if [ $? -ne 0 ]
# then
#     exit
# fi

python train_cnn.py
if [ $? -ne 0 ]
then
    exit
fi