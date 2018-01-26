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
"""Classes representing statistical distributions and ops for working with them.

See the @{$python/contrib.distributions} guide.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.contrib.distributions.python.ops.autoregressive import *
from tensorflow.contrib.distributions.python.ops.binomial import *
from tensorflow.contrib.distributions.python.ops.cauchy import *
from tensorflow.contrib.distributions.python.ops.chi2 import *
from tensorflow.contrib.distributions.python.ops.conditional_distribution import *
from tensorflow.contrib.distributions.python.ops.conditional_transformed_distribution import *
from tensorflow.contrib.distributions.python.ops.deterministic import *
from tensorflow.contrib.distributions.python.ops.distribution_util import fill_triangular
from tensorflow.contrib.distributions.python.ops.distribution_util import matrix_diag_transform
from tensorflow.contrib.distributions.python.ops.distribution_util import reduce_weighted_logsumexp
from tensorflow.contrib.distributions.python.ops.distribution_util import softplus_inverse
from tensorflow.contrib.distributions.python.ops.distribution_util import tridiag
from tensorflow.contrib.distributions.python.ops.estimator import *
from tensorflow.contrib.distributions.python.ops.geometric import *
from tensorflow.contrib.distributions.python.ops.half_normal import *
from tensorflow.contrib.distributions.python.ops.independent import *
from tensorflow.contrib.distributions.python.ops.inverse_gamma import *
from tensorflow.contrib.distributions.python.ops.logistic import *
from tensorflow.contrib.distributions.python.ops.mixture import *
from tensorflow.contrib.distributions.python.ops.mixture_same_family import *
from tensorflow.contrib.distributions.python.ops.moving_stats import *
from tensorflow.contrib.distributions.python.ops.mvn_diag import *
from tensorflow.contrib.distributions.python.ops.mvn_diag_plus_low_rank import *
from tensorflow.contrib.distributions.python.ops.mvn_full_covariance import *
from tensorflow.contrib.distributions.python.ops.mvn_tril import *
from tensorflow.contrib.distributions.python.ops.negative_binomial import *
from tensorflow.contrib.distributions.python.ops.normal_conjugate_posteriors import *
from tensorflow.contrib.distributions.python.ops.onehot_categorical import *
from tensorflow.contrib.distributions.python.ops.poisson import *
from tensorflow.contrib.distributions.python.ops.poisson_lognormal import *
from tensorflow.contrib.distributions.python.ops.quantized_distribution import *
from tensorflow.contrib.distributions.python.ops.relaxed_bernoulli import *
from tensorflow.contrib.distributions.python.ops.relaxed_onehot_categorical import *
from tensorflow.contrib.distributions.python.ops.sample_stats import *
from tensorflow.contrib.distributions.python.ops.sinh_arcsinh import *
from tensorflow.contrib.distributions.python.ops.test_util import *
from tensorflow.contrib.distributions.python.ops.vector_diffeomixture import *
from tensorflow.contrib.distributions.python.ops.vector_exponential_diag import *
from tensorflow.contrib.distributions.python.ops.vector_laplace_diag import *
from tensorflow.contrib.distributions.python.ops.vector_sinh_arcsinh_diag import *
from tensorflow.contrib.distributions.python.ops.wishart import *
from tensorflow.python.ops.distributions.bernoulli import *
from tensorflow.python.ops.distributions.beta import *
from tensorflow.python.ops.distributions.categorical import *
from tensorflow.python.ops.distributions.dirichlet import *
from tensorflow.python.ops.distributions.dirichlet_multinomial import *
from tensorflow.python.ops.distributions.distribution import *
from tensorflow.python.ops.distributions.exponential import *
from tensorflow.python.ops.distributions.gamma import *
from tensorflow.python.ops.distributions.kullback_leibler import *
from tensorflow.python.ops.distributions.laplace import *
from tensorflow.python.ops.distributions.multinomial import *
from tensorflow.python.ops.distributions.normal import *
from tensorflow.python.ops.distributions.student_t import *
from tensorflow.python.ops.distributions.transformed_distribution import *
from tensorflow.python.ops.distributions.uniform import *

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'auto_correlation',
    'bijectors',
    'Cauchy',
    'ConditionalDistribution',
    'ConditionalTransformedDistribution',
    'FULLY_REPARAMETERIZED',
    'NOT_REPARAMETERIZED',
    'ReparameterizationType',
    'Distribution',
    'Autoregressive',
    'Binomial',
    'Bernoulli',
    'BernoulliWithSigmoidProbs',
    'Beta',
    'BetaWithSoftplusConcentration',
    'Categorical',
    'Chi2',
    'Chi2WithAbsDf',
    'Deterministic',
    'VectorDeterministic',
    'Exponential',
    'ExponentialWithSoftplusRate',
    'VectorExponentialDiag',
    'Gamma',
    'GammaWithSoftplusConcentrationRate',
    'Geometric',
    'HalfNormal',
    'Independent',
    'InverseGamma',
    'InverseGammaWithSoftplusConcentrationRate',
    'Laplace',
    'LaplaceWithSoftplusScale',
    'Logistic',
    'NegativeBinomial',
    'Normal',
    'NormalWithSoftplusScale',
    'Poisson',
    'PoissonLogNormalQuadratureCompound',
    'SinhArcsinh',
    'StudentT',
    'StudentTWithAbsDfSoftplusScale',
    'Uniform',
    'MultivariateNormalDiag',
    'MultivariateNormalFullCovariance',
    'MultivariateNormalTriL',
    'MultivariateNormalDiagPlusLowRank',
    'MultivariateNormalDiagWithSoftplusScale',
    'Dirichlet',
    'DirichletMultinomial',
    'Multinomial',
    'VectorDiffeomixture',
    'VectorLaplaceDiag',
    'VectorSinhArcsinhDiag',
    'WishartCholesky',
    'WishartFull',
    'TransformedDistribution',
    'QuantizedDistribution',
    'Mixture',
    'MixtureSameFamily',
    'ExpRelaxedOneHotCategorical',
    'OneHotCategorical',
    'RelaxedBernoulli',
    'RelaxedOneHotCategorical',
    'kl_divergence',
    'RegisterKL',
    'fill_triangular',
    'matrix_diag_transform',
    'reduce_weighted_logsumexp',
    'softplus_inverse',
    'tridiag',
    'normal_conjugates_known_scale_posterior',
    'normal_conjugates_known_scale_predictive',
    'percentile',
    'assign_moving_mean_variance',
    'assign_log_moving_mean_exp',
    'moving_mean_variance',
    'estimator_head_distribution_regression',
    'quadrature_scheme_softmaxnormal_gauss_hermite',
    'quadrature_scheme_softmaxnormal_quantiles',
    'quadrature_scheme_lognormal_gauss_hermite',
    'quadrature_scheme_lognormal_quantiles',
]

remove_undocumented(__name__, _allowed_symbols)
