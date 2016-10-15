#!/bin/bash

apt-get update -y
apt-get install python-dev -y

pip install -U pillow
pip install -U numpy
pip install -U h5py
pip install -U theano
pip install -U keras

python -c "import keras" 2>/dev/null

sed -i -e s/\"tf\"/\"th\"/ ~/.keras/keras.json
sed -i -e s/\"tensorflow\"/\"theano\"/ ~/.keras/keras.json

