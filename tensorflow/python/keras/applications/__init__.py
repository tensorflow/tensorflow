# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras Applications are canned architectures with pre-trained weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.densenet import DenseNet169
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.nasnet import NASNetLarge
from tensorflow.python.keras.applications.nasnet import NASNetMobile
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.xception import Xception

del absolute_import
del division
del print_function
