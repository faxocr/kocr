#!/bin/bash

apt-get update -y
apt-get install python-dev libblas-dev -y

pip install Pillow==5.0.0
pip install numpy==1.14.1
pip install scipy==1.0.0
pip install h5py==2.7.1
pip install Theano==1.0.1
pip install Keras==2.1.4

python -c "import keras" 2>/dev/null

sed -i -e s/\"channels_last\"/\"channels_first\"/ ~/.keras/keras.json
sed -i -e s/\"tensorflow\"/\"theano\"/ ~/.keras/keras.json
