#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn import datasets

# Load Iris Data
iris = datasets.load_iris()

# Initialize a deep neural network autoencoder
# You can also add noise and add dropout if needed
# Details see TensorFlowDNNAutoencoder documentation.
autoencoder = learn.TensorFlowDNNAutoencoder(hidden_units=[10, 20])

# Fit with Iris data
transformed = autoencoder.fit_transform(iris.data)

print(transformed)
