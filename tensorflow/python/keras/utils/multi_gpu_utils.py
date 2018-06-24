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
"""Utilities for multi-gpu training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import tf_export


def _get_available_devices():
  return [x.name for x in K.get_session().list_devices()]


def _normalize_device_name(name):
  name = '/' + name.lower().split('device:')[1]
  return name


@tf_export('keras.utils.multi_gpu_model')
def multi_gpu_model(model, gpus, cpu_merge=True, cpu_relocation=False):
  """Replicates a model on different GPUs.

  Specifically, this function implements single-machine
  multi-GPU data parallelism. It works in the following way:

  - Divide the model's input(s) into multiple sub-batches.
  - Apply a model copy on each sub-batch. Every model copy
      is executed on a dedicated GPU.
  - Concatenate the results (on CPU) into one big batch.

  E.g. if your `batch_size` is 64 and you use `gpus=2`,
  then we will divide the input into 2 sub-batches of 32 samples,
  process each sub-batch on one GPU, then return the full
  batch of 64 processed samples.

  This induces quasi-linear speedup on up to 8 GPUs.

  This function is only available with the TensorFlow backend
  for the time being.

  Arguments:
      model: A Keras model instance. To avoid OOM errors,
          this model could have been built on CPU, for instance
          (see usage example below).
      gpus: Integer >= 2, number of on GPUs on which to create
          model replicas.
      cpu_merge: A boolean value to identify whether to force
          merging model weights under the scope of the CPU or not.
      cpu_relocation: A boolean value to identify whether to
          create the model's weights under the scope of the CPU.
          If the model is not defined under any preceding device
          scope, you can still rescue it by activating this option.

  Returns:
      A Keras `Model` instance which can be used just like the initial
      `model` argument, but which distributes its workload on multiple GPUs.

  Example 1: Training models with weights merge on CPU

  ```python
      import tensorflow as tf
      from keras.applications import Xception
      from keras.utils import multi_gpu_model
      import numpy as np

      num_samples = 1000
      height = 224
      width = 224
      num_classes = 1000

      # Instantiate the base model (or "template" model).
      # We recommend doing this with under a CPU device scope,
      # so that the model's weights are hosted on CPU memory.
      # Otherwise they may end up hosted on a GPU, which would
      # complicate weight sharing.
      with tf.device('/cpu:0'):
          model = Xception(weights=None,
                           input_shape=(height, width, 3),
                           classes=num_classes)

      # Replicates the model on 8 GPUs.
      # This assumes that your machine has 8 available GPUs.
      parallel_model = multi_gpu_model(model, gpus=8)
      parallel_model.compile(loss='categorical_crossentropy',
                             optimizer='rmsprop')

      # Generate dummy data.
      x = np.random.random((num_samples, height, width, 3))
      y = np.random.random((num_samples, num_classes))

      # This `fit` call will be distributed on 8 GPUs.
      # Since the batch size is 256, each GPU will process 32 samples.
      parallel_model.fit(x, y, epochs=20, batch_size=256)

      # Save model via the template model (which shares the same weights):
      model.save('my_model.h5')
  ```

  Example 2: Training models with weights merge on CPU using cpu_relocation

  ```python
       ..
       # Not needed to change the device scope for model definition:
       model = Xception(weights=None, ..)

       try:
           model = multi_gpu_model(model, cpu_relocation=True)
           print("Training using multiple GPUs..")
       except:
           print("Training using single GPU or CPU..")

       model.compile(..)
       ..
  ```

  Example 3: Training models with weights merge on GPU (recommended for NV-link)

  ```python
       ..
       # Not needed to change the device scope for model definition:
       model = Xception(weights=None, ..)

       try:
           model = multi_gpu_model(model, cpu_merge=False)
           print("Training using multiple GPUs..")
       except:
           print("Training using single GPU or CPU..")
       model.compile(..)
       ..
  ```

  Raises:
    ValueError: if the `gpus` argument does not match available devices.
  """
  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.layers.core import Lambda
  from tensorflow.python.keras.layers.merge import concatenate

  if isinstance(gpus, (list, tuple)):
    if len(gpus) <= 1:
      raise ValueError('For multi-gpu usage to be effective, '
                       'call `multi_gpu_model` with `len(gpus) >= 2`. '
                       'Received: `gpus=%s`' % gpus)
    num_gpus = len(gpus)
    target_gpu_ids = gpus
  else:
    if gpus <= 1:
      raise ValueError('For multi-gpu usage to be effective, '
                       'call `multi_gpu_model` with `gpus >= 2`. '
                       'Received: `gpus=%s`' % gpus)
    num_gpus = gpus
    target_gpu_ids = range(num_gpus)

  target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in target_gpu_ids]
  available_devices = _get_available_devices()
  available_devices = [
      _normalize_device_name(name) for name in available_devices
  ]
  for device in target_devices:
    if device not in available_devices:
      raise ValueError('To call `multi_gpu_model` with `gpus=%s`, '
                       'we expect the following devices to be available: %s. '
                       'However this machine only has: %s. '
                       'Try reducing `gpus`.' % (gpus, target_devices,
                                                 available_devices))

  def get_slice(data, i, parts):
    """Slice an array into `parts` slices and return slice `i`.

    Arguments:
      data: array to slice.
      i: index of slice to return.
      parts: number of slices to make.

    Returns:
      Slice `i` of `data`.
    """
    shape = array_ops.shape(data)
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == num_gpus - 1:
      size = batch_size - step * i
    else:
      size = step
    size = array_ops.concat([size, input_shape], axis=0)
    stride = array_ops.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return array_ops.slice(data, start, size)

  # Relocate the model definition under CPU device scope if needed
  if cpu_relocation:
    from tensorflow.python.keras.models import clone_model  # pylint: disable=g-import-not-at-top
    with ops.device('/cpu:0'):
      model = clone_model(model)

  all_outputs = []
  for i in range(len(model.outputs)):
    all_outputs.append([])

  # Place a copy of the model on each GPU,
  # each getting a slice of the inputs.
  for i, gpu_id in enumerate(target_gpu_ids):
    with ops.device('/gpu:%d' % gpu_id):
      with ops.name_scope('replica_%d' % gpu_id):
        inputs = []
        # Retrieve a slice of the input.
        for x in model.inputs:
          input_shape = tuple(x.get_shape().as_list())[1:]
          slice_i = Lambda(
              get_slice,
              output_shape=input_shape,
              arguments={
                  'i': i,
                  'parts': num_gpus
              })(
                  x)
          inputs.append(slice_i)

        # Apply model on slice
        # (creating a model replica on the target device).
        outputs = model(inputs)
        if not isinstance(outputs, list):
          outputs = [outputs]

        # Save the outputs for merging back together later.
        for o in range(len(outputs)):
          all_outputs[o].append(outputs[o])

  # Merge outputs under expected scope.
  with ops.device('/cpu:0' if cpu_merge else '/gpu:%d' % target_gpu_ids[0]):
    merged = []
    for name, outputs in zip(model.output_names, all_outputs):
      merged.append(concatenate(outputs, axis=0, name=name))
    return Model(model.inputs, merged)
