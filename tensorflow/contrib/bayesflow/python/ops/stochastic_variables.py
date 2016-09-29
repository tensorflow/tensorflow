# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Custom `get_variable` for stochastic variables.

@@get_stochastic_variable
@@make_stochastic_variable_getter
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib.bayesflow.python.ops import stochastic_tensor as st
from tensorflow.contrib.bayesflow.python.ops import variational_inference as vi
from tensorflow.contrib.distributions.python.ops import normal


def get_stochastic_variable(getter,
                            name,
                            shape=None,
                            dist_cls=normal.NormalWithSoftplusSigma,
                            dist_kwargs=None,
                            param_initializers=None,
                            prior=None,
                            **kwargs):
  """Custom variable getter for stochastic variables.

  `get_stochastic_variable` will create variables backing the parameters of a
  distribution, defined by `dist_cls`, and return a `StochasticTensor` which
  represents a sample from the backing distribution.

  Meant to be passed as the `custom_getter` to a `variable_scope`. Use
  `make_stochastic_variable_getter` to partially apply distribution-related
  args.

  Usage:

  ```python

  sv = tf.contrib.bayesflow.stochastic_variables
  dist = tf.contrib.distributions

  with tf.variable_scope('my_scope',
                         custom_getter=sv.make_stochastic_variable_getter(
                             dist_cls=dist.NormalWithSoftplusSigma
                             param_initializers={
                               "sigma": lambda shape, dtype, pi: (
                                   tf.constant(0.5, dtype=dtype, shape=shape))
                             })):
    v = tf.get_variable('my_var', (10, 20))
  ```

  `v` is a `StochasticTensor`, which is a sample from a backing
  `NormalWithSoftplusSigma` distribution. Underneath, 2 variables have been
  created: `my_var_mu` and `my_var_sigma`. `my_var_sigma` has been appropriately
  constrained to be positive by the `NormalWithSoftplusSigma` constructor, and
  initialized to a value of 0.5, which results in a sigma of ~1 after the
  softplus. The sample will have shape `(10, 20)`.

  Args:
    getter: original variable getter.
    name: prefix for variable(s) backing distribution parameters.
    shape: shape of the sample from the distribution (i.e. shape of the
        returned `StochasticTensor`).
    dist_cls: subclass of `Distribution` that implements `param_shapes`. Should
        accept unconstrained parameters (e.g. `NormalWithSoftplusSigma` accepts
        real-valued `sigma` and constrains it to be positive with `softplus`).
    dist_kwargs: `dict` of kwargs to be forwarded to `dist_cls`.
    param_initializers: `dict` from parameter name to initializer (see
        `get_variable` for initializer docs). Will override `initializer` in
        `kwargs`. `param_initializers` may contain initializers for only some of
        the parameters. Those parameters that do not contain entries will be
        initialized by `kwargs['initializer']`, if provided; otherwise, the
        default initialization of `getter` will be used.
    prior: instance of `Distribution` or a callable
        `(TensorShape, dtype) => Distribution`. If provided, will be registered
        as the prior for the `StochasticTensor` using
        `variational_inference.register_prior`.
    **kwargs: kwargs forwarded to `getter`.

  Returns:
    `StochasticTensor`, which represents a sample from the backing distribution.
  """
  param_initializers = param_initializers or {}
  param_shapes = {}

  if shape is not None:
    param_shapes = dist_cls.param_static_shapes(shape)

  param_names = set(list(param_shapes.keys()) + list(param_initializers.keys()))
  params = {}
  for param_name in param_names:
    # For each parameter, its param_initializer is used, if provided. Otherwise,
    # kwargs['initializer'] is used. If neither were provided, the default
    # variable initialization in getter will be used (i.e. getter will be passed
    # initializer=None.
    original_initializer = kwargs.pop('initializer', None)
    param_initializer = param_initializers.get(param_name, None)
    if param_initializer is None:
      param_initializer = original_initializer

    if callable(param_initializer) or param_initializer is None:
      param_shape = param_shapes.get(param_name, None)
    else:
      param_shape = None

    params[param_name] = getter(
        name + '_' + param_name,
        shape=param_shape,
        initializer=param_initializer,
        **kwargs)

  dist_kwargs = dist_kwargs or {}
  dist_kwargs.update(params)
  sample = st.StochasticTensor(dist_cls, **dist_kwargs)

  if prior is not None:
    if callable(prior):
      sample_value = sample.value()
      sample_value.get_shape().assert_is_fully_defined()
      prior = prior(sample_value.get_shape(), sample_value.dtype)

    vi.register_prior(sample, prior)

  return sample


def make_stochastic_variable_getter(dist_cls,
                                    dist_kwargs=None,
                                    param_initializers=None,
                                    prior=None):
  """`get_stochastic_variable` with args partially applied."""
  return functools.partial(
      get_stochastic_variable,
      dist_cls=dist_cls,
      dist_kwargs=dist_kwargs,
      param_initializers=param_initializers,
      prior=prior)
