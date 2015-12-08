"""TensorFlow ops for array / tensor manipulation."""

#  Copyright 2015 Google Inc. All Rights Reserved.
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


import tensorflow as tf


def split_squeeze(inp, dim):
    """Splits input on given dimenstion and then squeezes that dimension."""
    return [tf.squeeze_dim(t, dim) for t in tf.split(inp, dim)]


def expand_concat(inputs, dim):
    """Expands inputs on given dimension and then concats them."""
    return tf.concat(dim, [tf.expand_dim(t, dim) for t in inputs])
