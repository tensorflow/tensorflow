# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for Keras applications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import time

import six

from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.applications import efficientnet
from tensorflow.python.keras.applications import inception_resnet_v2
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.applications import nasnet
from tensorflow.python.keras.applications import resnet_v2
from tensorflow.python.keras.applications import vgg19
from tensorflow.python.keras.applications import xception
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class BenchmarkSaveApplications(
    six.with_metaclass(benchmark.ParameterizedBenchmark, test.Benchmark)):

  _benchmark_parameters = [
      ('ResNet152V2', resnet_v2.ResNet152V2, 2048),
      ('VGG19', vgg19.VGG19, 512),
      ('Xception', xception.Xception, 2048),
      ('InceptionResNetV2', inception_resnet_v2.InceptionResNetV2, 1536),
      ('MobileNetV2', mobilenet_v2.MobileNetV2, 1280),
      ('DenseNet201', densenet.DenseNet201, 1920),
      ('EfficientNetB7', efficientnet.EfficientNetB7, 2560),
      ('NASNetLarge', nasnet.NASNetLarge, 4032),
  ]

  def benchmark_save_and_load_applications(self, app, _):
    trials = 3

    model = app(weights=None)
    model_name = app.__name__

    tmp_dir = googletest.GetTempDir()
    gfile.MakeDirs(tmp_dir)
    save_dir = tempfile.mkdtemp(dir=tmp_dir)

    total_save_time = 0
    total_load_time = 0

    # Run one untimed iteration of saving/loading.
    model.save(save_dir, save_format='tf')
    keras_load.load(save_dir)

    for _ in range(trials):
      start_time = time.time()
      model.save(save_dir, save_format='tf')
      total_save_time += time.time() - start_time

      start_time = time.time()
      keras_load.load(save_dir)
      total_load_time += time.time() - start_time
    self.report_benchmark(
        iters=trials,
        wall_time=total_save_time / trials,
        name='{}.save'.format(model_name))

    self.report_benchmark(
        iters=1,
        wall_time=total_load_time / trials,
        name='{}.load'.format(model_name))
    gfile.DeleteRecursively(save_dir)


if __name__ == '__main__':
  test.main()
