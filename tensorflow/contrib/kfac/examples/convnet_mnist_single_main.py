# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Train a ConvNet on MNIST using K-FAC.

Train on single machine. See `convnet.train_mnist_single_machine` for details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
import tensorflow as tf

from tensorflow.contrib.kfac.examples import convnet

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "/tmp/mnist", "local mnist dir")


def main(unused_argv):
  convnet.train_mnist_single_machine(FLAGS.data_dir, num_epochs=200)


if __name__ == "__main__":
  tf.app.run(main=main)
