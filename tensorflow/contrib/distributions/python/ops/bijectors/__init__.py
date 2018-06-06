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
"""Bijector Ops.

@@AbsoluteValue
@@Affine
@@AffineLinearOperator
@@AffineScalar
@@Bijector
@@BatchNormalization
@@Chain
@@CholeskyOuterProduct
@@ConditionalBijector
@@Exp
@@Gumbel
@@Identity
@@Inline
@@Invert
@@Kumaraswamy
@@MaskedAutoregressiveFlow
@@MatrixInverseTriL
@@Ordered
@@Permute
@@PowerTransform
@@RealNVP
@@Reshape
@@Sigmoid
@@SinhArcsinh
@@SoftmaxCentered
@@Softplus
@@Softsign
@@Square
@@Weibull

@@masked_autoregressive_default_template
@@masked_dense
@@real_nvp_default_template
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.contrib.distributions.python.ops.bijectors.absolute_value import *
from tensorflow.contrib.distributions.python.ops.bijectors.affine import *
from tensorflow.contrib.distributions.python.ops.bijectors.affine_linear_operator import *
from tensorflow.contrib.distributions.python.ops.bijectors.affine_scalar import *
from tensorflow.contrib.distributions.python.ops.bijectors.batch_normalization import *
from tensorflow.contrib.distributions.python.ops.bijectors.chain import *
from tensorflow.contrib.distributions.python.ops.bijectors.cholesky_outer_product import *
from tensorflow.contrib.distributions.python.ops.bijectors.conditional_bijector import *
from tensorflow.contrib.distributions.python.ops.bijectors.exp import *
from tensorflow.contrib.distributions.python.ops.bijectors.gumbel import *
from tensorflow.contrib.distributions.python.ops.bijectors.inline import *
from tensorflow.contrib.distributions.python.ops.bijectors.invert import *
from tensorflow.contrib.distributions.python.ops.bijectors.kumaraswamy import *
from tensorflow.contrib.distributions.python.ops.bijectors.masked_autoregressive import *
from tensorflow.contrib.distributions.python.ops.bijectors.matrix_inverse_tril import *
from tensorflow.contrib.distributions.python.ops.bijectors.ordered import *
from tensorflow.contrib.distributions.python.ops.bijectors.permute import *
from tensorflow.contrib.distributions.python.ops.bijectors.power_transform import *
from tensorflow.contrib.distributions.python.ops.bijectors.real_nvp import *
from tensorflow.contrib.distributions.python.ops.bijectors.reshape import *
from tensorflow.contrib.distributions.python.ops.bijectors.sigmoid import *
from tensorflow.contrib.distributions.python.ops.bijectors.sinh_arcsinh import *
from tensorflow.contrib.distributions.python.ops.bijectors.softmax_centered import *
from tensorflow.contrib.distributions.python.ops.bijectors.softplus import *
from tensorflow.contrib.distributions.python.ops.bijectors.softsign import *
from tensorflow.contrib.distributions.python.ops.bijectors.square import *
from tensorflow.python.ops.distributions.bijector import *
from tensorflow.python.ops.distributions.identity_bijector import Identity

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.python.util.all_util import remove_undocumented

remove_undocumented(__name__)
