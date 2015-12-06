#!/bin/bash

# Create virtual env using system numpy and scipy
deactivate
virtualenv --system-site-packages testenv
source testenv/bin/activate

# Install dependencies
sudo pip install --upgrade pip
sudo pip install numpy
sudo pip install scipy
sudo pip install pandas
sudo pip install scikit-learn

# Install TensorFlow
if [ ${TRAVIS_OS_NAME} == "linux" ]; then
    sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
fi
if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    sudo pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
fi

# Install test tools
sudo pip install nose

# Install skflow
sudo python setup.py install

