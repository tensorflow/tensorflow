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

Multi tower training mode. See `convnet.train_mnist_multitower` for details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
import tensorflow as tf

from tensorflow.contrib.kfac.examples import convnet

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "/tmp/multitower_1/mnist", "local mnist dir")
flags.DEFINE_integer("num_towers", 2,
                     "Number of towers for multi tower training.")


def main(unused_argv):
  _ = unused_argv
  assert FLAGS.num_towers > 1
  devices = ["/gpu:{}".format(tower_id) for tower_id in range(FLAGS.num_towers)]
  convnet.train_mnist_multitower(
      FLAGS.data_dir,
      num_epochs=200,
      num_towers=FLAGS.num_towers,
      devices=devices)


if __name__ == "__main__":
  tf.app.run(main=main)
