#!/bin/bash

PIP=${PIP:-pip}
PYTHON=${PYTHON:-python}

apt-get update -y
apt-get install python-dev libblas-dev -y

"$PIP" install Pillow==5.0.0
"$PIP" install numpy==1.14.1
"$PIP" install scipy==1.0.0
"$PIP" install h5py==2.7.1
"$PIP" install Theano==1.0.1
"$PIP" install Keras==2.1.4

"$PYTHON" -c "import keras" 2>/dev/null

sed -i -e s/\"channels_last\"/\"channels_first\"/ ~/.keras/keras.json
sed -i -e s/\"tensorflow\"/\"theano\"/ ~/.keras/keras.json
