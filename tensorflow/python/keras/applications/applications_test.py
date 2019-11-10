# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for Keras applications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.applications import inception_resnet_v2
from tensorflow.python.keras.applications import inception_v3
from tensorflow.python.keras.applications import mobilenet
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.applications import nasnet
from tensorflow.python.keras.applications import resnet
from tensorflow.python.keras.applications import resnet_v2
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.applications import vgg19
from tensorflow.python.keras.applications import xception
from tensorflow.python.platform import test


MODEL_LIST = [
    (resnet.ResNet50, 2048),
    (resnet.ResNet101, 2048),
    (resnet.ResNet152, 2048),
    (resnet_v2.ResNet50V2, 2048),
    (resnet_v2.ResNet101V2, 2048),
    (resnet_v2.ResNet152V2, 2048),
    (vgg16.VGG16, 512),
    (vgg19.VGG19, 512),
    (xception.Xception, 2048),
    (inception_v3.InceptionV3, 2048),
    (inception_resnet_v2.InceptionResNetV2, 1536),
    (mobilenet.MobileNet, 1024),
    (mobilenet_v2.MobileNetV2, 1280),
    (densenet.DenseNet121, 1024),
    (densenet.DenseNet169, 1664),
    (densenet.DenseNet201, 1920),
    (nasnet.NASNetMobile, 1056),
]


class ApplicationsTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(*MODEL_LIST)
  def test_feature_extration_model(self, model_fn, output_dim):
    model = model_fn(include_top=False, weights=None)
    self.assertLen(model.output_shape, 4)
    self.assertEqual(model.output_shape[-1], output_dim)


if __name__ == '__main__':
  test.main()
