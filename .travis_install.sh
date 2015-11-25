#!/bin/bash

# Create virtual env
deactivate
virtualenv --system-site-packages testenv
source testenv/bin/activate

# Install dependencies
pip install sklearn scipy numpy
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# Install test tools
pip install nose

# Install skflow
python setup.py install

