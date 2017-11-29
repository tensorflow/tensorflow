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
"""Csiszar f-Divergence and helpers.

@@amari_alpha
@@arithmetic_geometric
@@chi_square
@@csiszar_vimco
@@dual_csiszar_function
@@jeffreys
@@jensen_shannon
@@kl_forward
@@kl_reverse
@@log1p_abs
@@modified_gan
@@monte_carlo_csiszar_f_divergence
@@pearson
@@squared_hellinger
@@symmetrized_csiszar_function
@@total_variation
@@triangular

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib.bayesflow.python.ops import monte_carlo_impl as monte_carlo
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util


def amari_alpha(logu, alpha=1., self_normalized=False, name=None):
  """The Amari-alpha Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True`, the Amari-alpha Csiszar-function is:

  ```none
  f(u) = { -log(u) + (u - 1),     alpha = 0
         { u log(u) - (u - 1),    alpha = 1
         { [(u**alpha - 1) - alpha (u - 1)] / (alpha (alpha - 1)),    otherwise
  ```

  When `self_normalized = False` the `(u - 1)` terms are omitted.

  Warning: when `alpha != 0` and/or `self_normalized = True` this function makes
  non-log-space calculations and may therefore be numerically unstable for
  `|logu| >> 0`.

  For more information, see:
    A. Cichocki and S. Amari. "Families of Alpha-Beta-and GammaDivergences:
    Flexible and Robust Measures of Similarities." Entropy, vol. 12, no. 6, pp.
    1532-1568, 2010.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    alpha: `float`-like Python scalar. (See Mathematical Details for meaning.)
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    amari_alpha_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.

  Raises:
    TypeError: if `alpha` is `None` or a `Tensor`.
    TypeError: if `self_normalized` is `None` or a `Tensor`.
  """
  with ops.name_scope(name, "amari_alpha", [logu]):
    if alpha is None or contrib_framework.is_tensor(alpha):
      raise TypeError("`alpha` cannot be `None` or `Tensor` type.")
    if self_normalized is None or contrib_framework.is_tensor(self_normalized):
      raise TypeError("`self_normalized` cannot be `None` or `Tensor` type.")

    logu = ops.convert_to_tensor(logu, name="logu")

    if alpha == 0.:
      f = -logu
    elif alpha == 1.:
      f = math_ops.exp(logu) * logu
    else:
      f = math_ops.expm1(alpha * logu) / (alpha * (alpha - 1.))

    if not self_normalized:
      return f

    if alpha == 0.:
      return f + math_ops.expm1(logu)
    elif alpha == 1.:
      return f - math_ops.expm1(logu)
    else:
      return f - math_ops.expm1(logu) / (alpha - 1.)


def kl_reverse(logu, self_normalized=False, name=None):
  """The reverse Kullback-Leibler Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True`, the KL-reverse Csiszar-function is:

  ```none
  f(u) = -log(u) + (u - 1)
  ```

  When `self_normalized = False` the `(u - 1)` term is omitted.

  Observe that as an f-Divergence, this Csiszar-function implies:

  ```none
  D_f[p, q] = KL[q, p]
  ```

  The KL is "reverse" because in maximum likelihood we think of minimizing `q`
  as in `KL[p, q]`.

  Warning: when self_normalized = True` this function makes non-log-space
  calculations and may therefore be numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    kl_reverse_of_u: `float`-like `Tensor` of the Csiszar-function evaluated at
      `u = exp(logu)`.

  Raises:
    TypeError: if `self_normalized` is `None` or a `Tensor`.
  """

  with ops.name_scope(name, "kl_reverse", [logu]):
    return amari_alpha(logu, alpha=0., self_normalized=self_normalized)


def kl_forward(logu, self_normalized=False, name=None):
  """The forward Kullback-Leibler Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True`, the KL-forward Csiszar-function is:

  ```none
  f(u) = u log(u) - (u - 1)
  ```

  When `self_normalized = False` the `(u - 1)` term is omitted.

  Observe that as an f-Divergence, this Csiszar-function implies:

  ```none
  D_f[p, q] = KL[p, q]
  ```

  The KL is "forward" because in maximum likelihood we think of minimizing `q`
  as in `KL[p, q]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    kl_forward_of_u: `float`-like `Tensor` of the Csiszar-function evaluated at
      `u = exp(logu)`.

  Raises:
    TypeError: if `self_normalized` is `None` or a `Tensor`.
  """

  with ops.name_scope(name, "kl_forward", [logu]):
    return amari_alpha(logu, alpha=1., self_normalized=self_normalized)


def jensen_shannon(logu, self_normalized=False, name=None):
  """The Jensen-Shannon Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True`, the Jensen-Shannon Csiszar-function is:

  ```none
  f(u) = u log(u) - (1 + u) log(1 + u) + (u + 1) log(2)
  ```

  When `self_normalized = False` the `(u + 1) log(2)` term is omitted.

  Observe that as an f-Divergence, this Csiszar-function implies:

  ```none
  D_f[p, q] = KL[p, m] + KL[q, m]
  m(x) = 0.5 p(x) + 0.5 q(x)
  ```

  In a sense, this divergence is the "reverse" of the Arithmetic-Geometric
  f-Divergence.

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  For more information, see:
    Lin, J. "Divergence measures based on the Shannon entropy." IEEE Trans.
    Inf. Th., 37, 145-151, 1991.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    jensen_shannon_of_u: `float`-like `Tensor` of the Csiszar-function
      evaluated at `u = exp(logu)`.
  """

  with ops.name_scope(name, "jensen_shannon", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    npdt = logu.dtype.as_numpy_dtype
    y = nn_ops.softplus(logu)
    if self_normalized:
      y -= np.log(2).astype(npdt)
    return math_ops.exp(logu) * logu - (1. + math_ops.exp(logu)) * y


def arithmetic_geometric(logu, self_normalized=False, name=None):
  """The Arithmetic-Geometric Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True` the Arithmetic-Geometric Csiszar-function is:

  ```none
  f(u) = (1 + u) log( (1 + u) / sqrt(u) ) - (1 + u) log(2)
  ```

  When `self_normalized = False` the `(1 + u) log(2)` term is omitted.

  Observe that as an f-Divergence, this Csiszar-function implies:

  ```none
  D_f[p, q] = KL[m, p] + KL[m, q]
  m(x) = 0.5 p(x) + 0.5 q(x)
  ```

  In a sense, this divergence is the "reverse" of the Jensen-Shannon
  f-Divergence.

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    arithmetic_geometric_of_u: `float`-like `Tensor` of the
      Csiszar-function evaluated at `u = exp(logu)`.
  """

  with ops.name_scope(name, "arithmetic_geometric", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    y = nn_ops.softplus(logu) - 0.5 * logu
    if self_normalized:
      y -= np.log(2.).astype(logu.dtype.as_numpy_dtype)
    return (1. + math_ops.exp(logu)) * y


def total_variation(logu, name=None):
  """The Total Variation Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Total-Variation Csiszar-function is:

  ```none
  f(u) = 0.5 |u - 1|
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    total_variation_of_u: `float`-like `Tensor` of the Csiszar-function
      evaluated at `u = exp(logu)`.
  """

  with ops.name_scope(name, "total_variation", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    return 0.5 * math_ops.abs(math_ops.expm1(logu))


def pearson(logu, name=None):
  """The Pearson Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Pearson Csiszar-function is:

  ```none
  f(u) = (u - 1)**2
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    pearson_of_u: `float`-like `Tensor` of the Csiszar-function evaluated at
      `u = exp(logu)`.
  """

  with ops.name_scope(name, "pearson", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    return math_ops.square(math_ops.expm1(logu))


def squared_hellinger(logu, name=None):
  """The Squared-Hellinger Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Squared-Hellinger Csiszar-function is:

  ```none
  f(u) = (sqrt(u) - 1)**2
  ```

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    squared_hellinger_of_u: `float`-like `Tensor` of the Csiszar-function
      evaluated at `u = exp(logu)`.
  """

  with ops.name_scope(name, "squared_hellinger", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    return pearson(0.5 * logu)


def triangular(logu, name=None):
  """The Triangular Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Triangular Csiszar-function is:

  ```none
  f(u) = (u - 1)**2 / (1 + u)
  ```

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    triangular_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with ops.name_scope(name, "triangular", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    return pearson(logu) / (1. + math_ops.exp(logu))


def t_power(logu, t, self_normalized=False, name=None):
  """The T-Power Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True` the T-Power Csiszar-function is:

  ```none
  f(u) = s [ u**t - 1 - t(u - 1) ]
  s = { -1   0 < t < 1
      { +1   otherwise
  ```

  When `self_normalized = False` the `- t(u - 1)` term is omitted.

  This is similar to the `amari_alpha` Csiszar-function, with the associated
  divergence being the same up to factors depending only on `t`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    t:  `Tensor` of same `dtype` as `logu` and broadcastable shape.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    t_power_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """
  with ops.name_scope(name, "t_power", [logu, t]):
    logu = ops.convert_to_tensor(logu, name="logu")
    t = ops.convert_to_tensor(t, dtype=logu.dtype.base_dtype, name="t")
    fu = math_ops.expm1(t * logu)
    if self_normalized:
      fu -= t * math_ops.expm1(logu)
    fu *= array_ops.where(math_ops.logical_and(0. < t, t < 1.),
                          -array_ops.ones_like(t),
                          array_ops.ones_like(t))
    return fu


def log1p_abs(logu, name=None):
  """The log1p-abs Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Log1p-Abs Csiszar-function is:

  ```none
  f(u) = u**(sign(u-1)) - 1
  ```

  This function is so-named because it was invented from the following recipe.
  Choose a convex function g such that g(0)=0 and solve for f:

  ```none
  log(1 + f(u)) = g(log(u)).
    <=>
  f(u) = exp(g(log(u))) - 1
  ```

  That is, the graph is identically `g` when y-axis is `log1p`-domain and x-axis
  is `log`-domain.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    log1p_abs_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with ops.name_scope(name, "log1p_abs", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    return math_ops.expm1(math_ops.abs(logu))


def jeffreys(logu, name=None):
  """The Jeffreys Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Jeffreys Csiszar-function is:

  ```none
  f(u) = 0.5 ( u log(u) - log(u) )
       = 0.5 kl_forward + 0.5 kl_reverse
       = symmetrized_csiszar_function(kl_reverse)
       = symmetrized_csiszar_function(kl_forward)
  ```

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    jeffreys_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with ops.name_scope(name, "jeffreys", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    return 0.5 * math_ops.expm1(logu) * logu


def chi_square(logu, name=None):
  """The chi-Square Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Chi-square Csiszar-function is:

  ```none
  f(u) = u**2 - 1
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    chi_square_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with ops.name_scope(name, "chi_square", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    return math_ops.expm1(2. * logu)


def modified_gan(logu, self_normalized=False, name=None):
  """The Modified-GAN Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True` the modified-GAN (Generative/Adversarial
  Network) Csiszar-function is:

  ```none
  f(u) = log(1 + u) - log(u) + 0.5 (u - 1)
  ```

  When `self_normalized = False` the `0.5 (u - 1)` is omitted.

  The unmodified GAN Csiszar-function is identical to Jensen-Shannon (with
  `self_normalized = False`).

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    chi_square_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with ops.name_scope(name, "chi_square", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    y = nn_ops.softplus(logu) - logu
    if self_normalized:
      y += 0.5 * math_ops.expm1(logu)
    return y


def dual_csiszar_function(logu, csiszar_function, name=None):
  """Calculates the dual Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Csiszar-dual is defined as:

  ```none
  f^*(u) = u f(1 / u)
  ```

  where `f` is some other Csiszar-function.

  For example, the dual of `kl_reverse` is `kl_forward`, i.e.,

  ```none
  f(u) = -log(u)
  f^*(u) = u f(1 / u) = -u log(1 / u) = u log(u)
  ```

  The dual of the dual is the original function:

  ```none
  f^**(u) = {u f(1/u)}^*(u) = u (1/u) f(1/(1/u)) = f(u)
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    csiszar_function: Python `callable` representing a Csiszar-function over
      log-domain.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    dual_f_of_u: `float`-like `Tensor` of the result of calculating the dual of
      `f` at `u = exp(logu)`.
  """

  with ops.name_scope(name, "dual_csiszar_function", [logu]):
    return math_ops.exp(logu) * csiszar_function(-logu)


def symmetrized_csiszar_function(logu, csiszar_function, name=None):
  """Symmetrizes a Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The symmetrized Csiszar-function is defined as:

  ```none
  f_g(u) = 0.5 g(u) + 0.5 u g (1 / u)
  ```

  where `g` is some other Csiszar-function.

  We say the function is "symmetrized" because:

  ```none
  D_{f_g}[p, q] = D_{f_g}[q, p]
  ```

  for all `p << >> q` (i.e., `support(p) = support(q)`).

  There exists alternatives for symmetrizing a Csiszar-function. For example,

  ```none
  f_g(u) = max(f(u), f^*(u)),
  ```

  where `f^*` is the dual Csiszar-function, also implies a symmetric
  f-Divergence.

  Example:

  When either of the following functions are symmetrized, we obtain the
  Jensen-Shannon Csiszar-function, i.e.,

  ```none
  g(u) = -log(u) - (1 + u) log((1 + u) / 2) + u - 1
  h(u) = log(4) + 2 u log(u / (1 + u))
  ```

  implies,

  ```none
  f_g(u) = f_h(u) = u log(u) - (1 + u) log((1 + u) / 2)
         = jensen_shannon(log(u)).
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    csiszar_function: Python `callable` representing a Csiszar-function over
      log-domain.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    symmetrized_g_of_u: `float`-like `Tensor` of the result of applying the
      symmetrization of `g` evaluated at `u = exp(logu)`.
  """

  with ops.name_scope(name, "symmetrized_csiszar_function", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")
    return 0.5 * (csiszar_function(logu)
                  + dual_csiszar_function(logu, csiszar_function))


def monte_carlo_csiszar_f_divergence(
    f,
    p_log_prob,
    q,
    num_draws,
    use_reparametrization=None,
    seed=None,
    name=None):
  """Monte-Carlo approximation of the Csiszar f-Divergence.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Csiszar f-Divergence for Csiszar-function f is given by:

  ```none
  D_f[p(X), q(X)] := E_{q(X)}[ f( p(X) / q(X) ) ]
                  ~= m**-1 sum_j^m f( p(x_j) / q(x_j) ),
                             where x_j ~iid q(X)
  ```

  Tricks: Reparameterization and Score-Gradient

  When q is "reparameterized", i.e., a diffeomorphic transformation of a
  parameterless distribution (e.g.,
  `Normal(Y; m, s) <=> Y = sX + m, X ~ Normal(0,1)`), we can swap gradient and
  expectation, i.e.,
  `grad[Avg{ s_i : i=1...n }] = Avg{ grad[s_i] : i=1...n }` where `S_n=Avg{s_i}`
  and `s_i = f(x_i), x_i ~iid q(X)`.

  However, if q is not reparameterized, TensorFlow's gradient will be incorrect
  since the chain-rule stops at samples of unreparameterized distributions. In
  this circumstance using the Score-Gradient trick results in an unbiased
  gradient, i.e.,

  ```none
  grad[ E_q[f(X)] ]
  = grad[ int dx q(x) f(x) ]
  = int dx grad[ q(x) f(x) ]
  = int dx [ q'(x) f(x) + q(x) f'(x) ]
  = int dx q(x) [q'(x) / q(x) f(x) + f'(x) ]
  = int dx q(x) grad[ f(x) q(x) / stop_grad[q(x)] ]
  = E_q[ grad[ f(x) q(x) / stop_grad[q(x)] ] ]
  ```

  Unless `q.reparameterization_type != distribution.FULLY_REPARAMETERIZED` it is
  usually preferable to set `use_reparametrization = True`.

  Example Application:

  The Csiszar f-Divergence is a useful framework for variational inference.
  I.e., observe that,

  ```none
  f(p(x)) =  f( E_{q(Z | x)}[ p(x, Z) / q(Z | x) ] )
          <= E_{q(Z | x)}[ f( p(x, Z) / q(Z | x) ) ]
          := D_f[p(x, Z), q(Z | x)]
  ```

  The inequality follows from the fact that the "perspective" of `f`, i.e.,
  `(s, t) |-> t f(s / t))`, is convex in `(s, t)` when `s/t in domain(f)` and
  `t` is a real. Since the above framework includes the popular Evidence Lower
  BOund (ELBO) as a special case, i.e., `f(u) = -log(u)`, we call this framework
  "Evidence Divergence Bound Optimization" (EDBO).

  Args:
    f: Python `callable` representing a Csiszar-function in log-space, i.e.,
      takes `p_log_prob(q_samples) - q.log_prob(q_samples)`.
    p_log_prob: Python `callable` taking (a batch of) samples from `q` and
      returning the natural-log of the probability under distribution `p`.
      (In variational inference `p` is the joint distribution.)
    q: `tf.Distribution`-like instance; must implement:
      `reparameterization_type`, `sample(n, seed)`, and `log_prob(x)`.
      (In variational inference `q` is the approximate posterior distribution.)
    num_draws: Integer scalar number of draws used to approximate the
      f-Divergence expectation.
    use_reparametrization: Python `bool`. When `None` (the default),
      automatically set to:
      `q.reparameterization_type == distribution.FULLY_REPARAMETERIZED`.
      When `True` uses the standard Monte-Carlo average. When `False` uses the
      score-gradient trick. (See above for details.)  When `False`, consider
      using `csiszar_vimco`.
    seed: Python `int` seed for `q.sample`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    monte_carlo_csiszar_f_divergence: `float`-like `Tensor` Monte Carlo
      approximation of the Csiszar f-Divergence.

  Raises:
    ValueError: if `q` is not a reparameterized distribution and
      `use_reparametrization = True`. A distribution `q` is said to be
      "reparameterized" when its samples are generated by transforming the
      samples of another distribution which does not depend on the
      parameterization of `q`. This property ensures the gradient (with respect
      to parameters) is valid.
    TypeError: if `p_log_prob` is not a Python `callable`.
  """
  with ops.name_scope(name, "monte_carlo_csiszar_f_divergence", [num_draws]):
    if use_reparametrization is None:
      use_reparametrization = (q.reparameterization_type
                               == distribution.FULLY_REPARAMETERIZED)
    elif (use_reparametrization and
          q.reparameterization_type != distribution.FULLY_REPARAMETERIZED):
      # TODO(jvdillon): Consider only raising an exception if the gradient is
      # requested.
      raise ValueError(
          "Distribution `q` must be reparameterized, i.e., a diffeomorphic "
          "transformation of a parameterless distribution. (Otherwise this "
          "function has a biased gradient.)")
    if not callable(p_log_prob):
      raise TypeError("`p_log_prob` must be a Python `callable` function.")
    return monte_carlo.expectation(
        f=lambda q_samples: f(p_log_prob(q_samples) - q.log_prob(q_samples)),
        samples=q.sample(num_draws, seed=seed),
        log_prob=q.log_prob,  # Only used if use_reparametrization=False.
        use_reparametrization=use_reparametrization)


def csiszar_vimco(f,
                  p_log_prob,
                  q,
                  num_draws,
                  num_batch_draws=1,
                  seed=None,
                  name=None):
  """Use VIMCO to lower the variance of gradient[csiszar_function(Avg(logu))].

  This function generalizes "Variational Inference for Monte Carlo Objectives"
  (VIMCO), i.e., https://arxiv.org/abs/1602.06725, to Csiszar f-Divergences.

  Note: if `q.reparameterization_type = distribution.FULLY_REPARAMETERIZED`,
  consider using `monte_carlo_csiszar_f_divergence`.

  The VIMCO loss is:

  ```none
  vimco = f(Avg{logu[i] : i=0,...,m-1})
  where,
    logu[i] = log( p(x, h[i]) / q(h[i] | x) )
    h[i] iid~ q(H | x)
  ```

  Interestingly, the VIMCO gradient is not the naive gradient of `vimco`.
  Rather, it is characterized by:

  ```none
  grad[vimco] - variance_reducing_term
  where,
    variance_reducing_term = Sum{ grad[log q(h[i] | x)] *
                                    (vimco - f(log Avg{h[j;i] : j=0,...,m-1}))
                                 : i=0, ..., m-1 }
    h[j;i] = { u[j]                             j!=i
             { GeometricAverage{ u[k] : k!=i}   j==i
  ```

  (We omitted `stop_gradient` for brevity. See implementation for more details.)

  The `Avg{h[j;i] : j}` term is a kind of "swap-out average" where the `i`-th
  element has been replaced by the leave-`i`-out Geometric-average.

  This implementation prefers numerical precision over efficiency, i.e.,
  `O(num_draws * num_batch_draws * prod(batch_shape) * prod(event_shape))`.
  (The constant may be fairly large, perhaps around 12.)

  Args:
    f: Python `callable` representing a Csiszar-function in log-space.
    p_log_prob: Python `callable` representing the natural-log of the
      probability under distribution `p`. (In variational inference `p` is the
      joint distribution.)
    q: `tf.Distribution`-like instance; must implement: `sample(n, seed)`, and
      `log_prob(x)`. (In variational inference `q` is the approximate posterior
      distribution.)
    num_draws: Integer scalar number of draws used to approximate the
      f-Divergence expectation.
    num_batch_draws: Integer scalar number of draws used to approximate the
      f-Divergence expectation.
    seed: Python `int` seed for `q.sample`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    vimco: The Csiszar f-Divergence generalized VIMCO objective.

  Raises:
    ValueError: if `num_draws < 2`.
  """
  with ops.name_scope(name, "csiszar_vimco", [num_draws, num_batch_draws]):
    if num_draws < 2:
      raise ValueError("Must specify num_draws > 1.")
    stop = array_ops.stop_gradient  # For readability.
    x = stop(q.sample(sample_shape=[num_draws, num_batch_draws],
                      seed=seed))
    logqx = q.log_prob(x)
    logu = p_log_prob(x) - logqx
    f_log_avg_u, f_log_sooavg_u = [f(r) for r in csiszar_vimco_helper(logu)]
    dotprod = math_ops.reduce_sum(
        logqx * stop(f_log_avg_u - f_log_sooavg_u),
        axis=0)  # Sum over iid samples.
    # We now rewrite f_log_avg_u so that:
    #   `grad[f_log_avg_u] := grad[f_log_avg_u + dotprod]`.
    # To achieve this, we use a trick that
    #   `f(x) - stop(f(x)) == zeros_like(f(x))`
    # but its gradient is grad[f(x)].
    # Note that IEEE754 specifies that `x - x == 0.` and `x + 0. == x`, hence
    # this trick loses no precision. For more discussion regarding the relevant
    # portions of the IEEE754 standard, see the StackOverflow question,
    # "Is there a floating point value of x, for which x-x == 0 is false?"
    # http://stackoverflow.com/q/2686644
    f_log_avg_u += dotprod - stop(dotprod)  # Add zeros_like(dot_prod).
    return math_ops.reduce_mean(f_log_avg_u, axis=0)  # Avg over batches.


def csiszar_vimco_helper(logu, name=None):
  """Helper to `csiszar_vimco`; computes `log_avg_u`, `log_sooavg_u`.

  `axis = 0` of `logu` is presumed to correspond to iid samples from `q`, i.e.,

  ```none
  logu[j] = log(u[j])
  u[j] = p(x, h[j]) / q(h[j] | x)
  h[j] iid~ q(H | x)
  ```

  Args:
    logu: Floating-type `Tensor` representing `log(p(x, h) / q(h | x))`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    log_avg_u: `logu.dtype` `Tensor` corresponding to the natural-log of the
      average of `u`. The sum of the gradient of `log_avg_u` is `1`.
    log_sooavg_u: `logu.dtype` `Tensor` characterized by the natural-log of the
      average of `u`` except that the average swaps-out `u[i]` for the
      leave-`i`-out Geometric-average. The mean of the gradient of
      `log_sooavg_u` is `1`. Mathematically `log_sooavg_u` is,
      ```none
      log_sooavg_u[i] = log(Avg{h[j ; i] : j=0, ..., m-1})
      h[j ; i] = { u[j]                              j!=i
                 { GeometricAverage{u[k] : k != i}   j==i
      ```

  """
  with ops.name_scope(name, "csiszar_vimco_helper", [logu]):
    logu = ops.convert_to_tensor(logu, name="logu")

    n = logu.shape.with_rank_at_least(1)[0].value
    if n is None:
      n = array_ops.shape(logu)[0]
      log_n = math_ops.log(math_ops.cast(n, dtype=logu.dtype))
      nm1 = math_ops.cast(n - 1, dtype=logu.dtype)
    else:
      log_n = np.log(n).astype(logu.dtype.as_numpy_dtype)
      nm1 = np.asarray(n - 1, dtype=logu.dtype.as_numpy_dtype)

    # Throughout we reduce across axis=0 since this is presumed to be iid
    # samples.

    log_max_u = math_ops.reduce_max(logu, axis=0)
    log_sum_u_minus_log_max_u = math_ops.reduce_logsumexp(
        logu - log_max_u, axis=0)

    # log_loosum_u[i] =
    # = logsumexp(logu[j] : j != i)
    # = log( exp(logsumexp(logu)) - exp(logu[i]) )
    # = log( exp(logsumexp(logu - logu[i])) exp(logu[i])  - exp(logu[i]))
    # = logu[i] + log(exp(logsumexp(logu - logu[i])) - 1)
    # = logu[i] + log(exp(logsumexp(logu) - logu[i]) - 1)
    # = logu[i] + softplus_inverse(logsumexp(logu) - logu[i])
    d = log_sum_u_minus_log_max_u + (log_max_u - logu)
    # We use `d != 0` rather than `d > 0.` because `d < 0.` should never
    # happens; if it does we want to complain loudly (which `softplus_inverse`
    # will).
    d_ok = math_ops.not_equal(d, 0.)
    safe_d = array_ops.where(d_ok, d, array_ops.ones_like(d))
    d_ok_result = logu + distribution_util.softplus_inverse(safe_d)

    inf = np.array(np.inf, dtype=logu.dtype.as_numpy_dtype)

    # When not(d_ok) and is_positive_and_largest then we manually compute the
    # log_loosum_u. (We can efficiently do this for any one point but not all,
    # hence we still need the above calculation.) This is good because when
    # this condition is met, we cannot use the above calculation; its -inf.
    # We now compute the log-leave-out-max-sum, replicate it to every
    # point and make sure to select it only when we need to.
    is_positive_and_largest = math_ops.logical_and(
        logu > 0.,
        math_ops.equal(logu, log_max_u[array_ops.newaxis, ...]))
    log_lomsum_u = math_ops.reduce_logsumexp(
        array_ops.where(is_positive_and_largest,
                        array_ops.fill(array_ops.shape(logu), -inf),
                        logu),
        axis=0, keep_dims=True)
    log_lomsum_u = array_ops.tile(
        log_lomsum_u,
        multiples=1 + array_ops.pad([n-1], [[0, array_ops.rank(logu)-1]]))

    d_not_ok_result = array_ops.where(
        is_positive_and_largest,
        log_lomsum_u,
        array_ops.fill(array_ops.shape(d), -inf))

    log_loosum_u = array_ops.where(d_ok, d_ok_result, d_not_ok_result)

    # The swap-one-out-sum ("soosum") is n different sums, each of which
    # replaces the i-th item with the i-th-left-out average, i.e.,
    # soo_sum_u[i] = [exp(logu) - exp(logu[i])] + exp(mean(logu[!=i]))
    #              =  exp(log_loosum_u[i])      + exp(looavg_logu[i])
    looavg_logu = (math_ops.reduce_sum(logu, axis=0) - logu) / nm1
    log_soosum_u = math_ops.reduce_logsumexp(
        array_ops.stack([log_loosum_u, looavg_logu]),
        axis=0)

    log_avg_u = log_sum_u_minus_log_max_u + log_max_u - log_n
    log_sooavg_u = log_soosum_u - log_n

    log_avg_u.set_shape(logu.shape.with_rank_at_least(1)[1:])
    log_sooavg_u.set_shape(logu.shape)

    return log_avg_u, log_sooavg_u
