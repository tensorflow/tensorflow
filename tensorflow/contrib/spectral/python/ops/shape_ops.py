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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def frames(signal, frame_length, frame_step, name="frames"):
  """Frame a signal into overlapping frames.
  May be used in front of spectral functions.
  
  For example:
  
  ```python
  pcm = tf.placeholder(tf.float32, [None, 9152])
  frames = tf.frames(pcm, 512, 180)
  magspec = tf.abs(tf.spectral.rfft(frames, tf.constant(512, shape=[1])))
  image = tf.reshape(magspec, [-1, 49, 257, 1])
  ```
  
  Args:
    signal: A `Tensor` of shape [batch_size, signal_length].
    frame_length: An `int32` or `int64` `Tensor`. The length of each frame.
    frame_step: An `int32` or `int64` `Tensor`. The step between frames.
    name: A name for the operation (optional).
  
  Returns:
    A `Tensor` of frames with shape [batch_size, num_frames, frame_length].
  """
  signal = ops.convert_to_tensor(signal)
  frame_length = ops.convert_to_tensor(frame_length)
  frame_step = ops.convert_to_tensor(frame_step)
  
  signal_rank = signal.shape.ndims
  
  if signal_rank != 2:
    raise ValueError("expected signal to have rank 2 but was " + signal_rank)
  
  signal_length = array_ops.shape(signal)[1]
  
  with ops.name_scope(name, "frames", [signal]) as name:
    num_frames = 1 + math_ops.cast(math_ops.ceil((signal_length - frame_length) / frame_step), dtype=dtypes.int32)
    pad_length = (num_frames - 1) * frame_step + frame_length
    pad_signal = array_ops.pad(signal, [[0, 0], [0, pad_length - signal_length]])
    
    indices_frames = array_ops.expand_dims(array_ops.tile(math_ops.range(0, frame_len), [num_frames]), 1)
    indices_steps = array_ops.expand_dims(array_ops.tile(math_ops.range(0, num_frames * frame_step, frame_step), [frame_len]), 0)
    indices = indices_frames + indices_steps
    
    return array_ops.gather(pad_signal, indices, name=name)
