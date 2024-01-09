# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Stochastic cast op which stochastically casts input tensors to the desired data type."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_stochastic_cast_op
from tensorflow.python.ops import random_ops_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


def allowed_to_types(is_integer=True):
  if is_integer:
    return {dtypes.int32, dtypes.int16, dtypes.int8}
  else:
    return {
        dtypes.float16,
        dtypes.bfloat16,
        dtypes.float8_e5m2,
        dtypes.float8_e4m3fn,
    }


@tf_export("random.stochastic_cast")
@dispatch.add_dispatch_support
def stochastic_cast(
    t,
    dtype,
    seed,
    alg="auto_select",
    name=None,
):
  """Casts input to the desired precision with stochastic rounding.

  This means the value of the cast result will be rounded to two of the closest
  values with with a probability proportional to the distance between the number
  and the two closest to the input. For example, if a number falls between 2 and
  3, and is closer to 2 than to 3, it has a higher probability of being rounded
  to 2. On the other hand, if it's closer to 3 than to 2, it has a higher
  probability of being rounded to 3. This is intended to eliminate rounding bias
  introduced by determinisitc rounding methods. If cast to integers, the values
  will saturate if out of range, e.g. 254.8 in floating point will become 127 in
  int8. If inputs are NaN, the results will be zero. Given the same random seed,
  the results will be deterministic, but not otherwise.

  Args:
    t: The input tensor. This is the same as the output shape.
    dtype: The output type, currently int32, int16 and int8 are supported.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    alg: The RNG algorithm used to generate the random numbers. See
      `tf.random.stateless_uniform` for a detailed explanation.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified data type whose values are rounded to the
    specified precisions with stochastic rounding.
  """
  with ops.name_scope(name, "stochastic_cast", [t, seed]) as name:
    t = ops.convert_to_tensor(t)
    key, counter, algorithm = random_ops_util.get_key_counter_alg(seed, alg)
    if dtype in allowed_to_types(is_integer=True):
      return gen_stochastic_cast_op.stochastic_cast_to_int(
          t, key=key, counter=counter, alg=algorithm, Tout=dtype
      )
    else:
      # TODO(b/232442915): Support casting to small floats.
      raise NotImplementedError(
          f"Stochastic cast to small float {dtype} has not yet been supported."
      )
