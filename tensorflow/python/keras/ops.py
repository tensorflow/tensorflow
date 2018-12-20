# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Module for exporting TensorFlow ops under tf.keras.*."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util.tf_export import keras_export


keras_export("keras.initializers.Initializer")(
    init_ops.Initializer)
keras_export("keras.initializers.Zeros", "keras.initializers.zeros")(
    init_ops.Zeros)
keras_export("keras.initializers.Ones", "keras.initializers.ones")(
    init_ops.Ones)
keras_export("keras.initializers.Constant", "keras.initializers.constant")(
    init_ops.Constant)
keras_export("keras.initializers.VarianceScaling")(
    init_ops.VarianceScaling)
keras_export("keras.initializers.Orthogonal", "keras.initializers.orthogonal")(
    init_ops.Orthogonal)
keras_export("keras.initializers.Identity", "keras.initializers.identity")(
    init_ops.Identity)
keras_export("keras.initializers.glorot_uniform")(
    init_ops.GlorotUniform)
keras_export("keras.initializers.glorot_normal")(
    init_ops.GlorotNormal)
keras_export("keras.initializers.lecun_normal")(
    init_ops.lecun_normal)
keras_export("keras.initializers.lecun_uniform")(
    init_ops.lecun_uniform)
keras_export("keras.initializers.he_normal")(
    init_ops.he_normal)
keras_export("keras.initializers.he_uniform")(
    init_ops.he_uniform)

keras_export("keras.backend.name_scope")(ops.name_scope)

keras_export("keras.losses.Reduction", v1=[])(
    losses_impl.ReductionV2)
