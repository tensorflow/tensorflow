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

from tensorflow.python.keras import applications
from tensorflow.python.platform import test


MODEL_LIST = [
    (applications.ResNet50, 2048),
    (applications.VGG16, 512),
    (applications.VGG19, 512),
    (applications.Xception, 2048),
    (applications.InceptionV3, 2048),
    (applications.InceptionResNetV2, 1536),
    (applications.MobileNet, 1024),
    # TODO(fchollet): enable MobileNetV2 tests when a new TensorFlow test image
    # is released with keras_applications upgraded to 1.0.5 or above.
    (applications.DenseNet121, 1024),
    (applications.DenseNet169, 1664),
    (applications.DenseNet201, 1920),
    (applications.NASNetMobile, 1056),
    (applications.NASNetLarge, 4032),
]


class ApplicationsTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(*MODEL_LIST)
  def test_feature_extration_model(self, model_fn, output_dim):
    model = model_fn(include_top=False, weights=None)
    self.assertEqual(model.output_shape, (None, None, None, output_dim))


if __name__ == '__main__':
  test.main()
