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
"""Loss functions to be used by LayerCollection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.contrib.distributions.python.ops import onehot_categorical
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.ops.distributions import normal


@six.add_metaclass(abc.ABCMeta)
class LossFunction(object):
  """Abstract base class for loss functions.

  Note that unlike typical loss functions used in neural networks these are
  summed and not averaged across cases in the batch, since this is what the
  users of this class (FisherEstimator and MatrixVectorProductComputer) will
  be expecting. The implication of this is that you will may want to
  normalize things like Fisher-vector products by the batch size when you
  use this class.  It depends on the use case.
  """

  @abc.abstractproperty
  def targets(self):
    """The targets being predicted by the model.

    Returns:
      None or Tensor of appropriate shape for calling self._evaluate() on.
    """
    pass

  @abc.abstractproperty
  def inputs(self):
    """The inputs to the loss function (excluding the targets)."""
    pass

  def evaluate(self):
    """Evaluate the loss function on the targets."""
    if self.targets is not None:
      # We treat the targets as "constant".  It's only the inputs that get
      # "back-propped" through.
      return self._evaluate(array_ops.stop_gradient(self.targets))
    else:
      raise Exception("Cannot evaluate losses with unspecified targets.")

  @abc.abstractmethod
  def _evaluate(self, targets):
    """Evaluates the negative log probability of the targets.

    Args:
      targets: Tensor that distribution can calculate log_prob() of.

    Returns:
      negative log probability of each target, summed across all targets.
    """
    pass

  @abc.abstractmethod
  def multiply_hessian(self, vector):
    """Right-multiply a vector by the Hessian.

    Here the 'Hessian' is the Hessian matrix (i.e. matrix of 2nd-derivatives)
    of the loss function with respect to its inputs.

    Args:
      vector: The vector to multiply.  Must be the same shape(s) as the
        'inputs' property.

    Returns:
      The vector right-multiplied by the Hessian.  Will be of the same shape(s)
      as the 'inputs' property.
    """
    pass

  @abc.abstractmethod
  def multiply_hessian_factor(self, vector):
    """Right-multiply a vector by a factor B of the Hessian.

    Here the 'Hessian' is the Hessian matrix (i.e. matrix of 2nd-derivatives)
    of the loss function with respect to its inputs.  Typically this will be
    block-diagonal across different cases in the batch, since the loss function
    is typically summed across cases.

    Note that B can be any matrix satisfying B * B^T = H where H is the Hessian,
    but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply.  Must be of the shape given by the
        'hessian_factor_inner_shape' property.

    Returns:
      The vector right-multiplied by B.  Will be of the same shape(s) as the
      'inputs' property.
    """
    pass

  @abc.abstractmethod
  def multiply_hessian_factor_transpose(self, vector):
    """Right-multiply a vector by the transpose of a factor B of the Hessian.

    Here the 'Hessian' is the Hessian matrix (i.e. matrix of 2nd-derivatives)
    of the loss function with respect to its inputs.  Typically this will be
    block-diagonal across different cases in the batch, since the loss function
    is typically summed across cases.

    Note that B can be any matrix satisfying B * B^T = H where H is the Hessian,
    but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply.  Must be the same shape(s) as the
        'inputs' property.

    Returns:
      The vector right-multiplied by B^T.  Will be of the shape given by the
      'hessian_factor_inner_shape' property.
    """
    pass

  @abc.abstractmethod
  def multiply_hessian_factor_replicated_one_hot(self, index):
    """Right-multiply a replicated-one-hot vector by a factor B of the Hessian.

    Here the 'Hessian' is the Hessian matrix (i.e. matrix of 2nd-derivatives)
    of the loss function with respect to its inputs.  Typically this will be
    block-diagonal across different cases in the batch, since the loss function
    is typically summed across cases.

    A 'replicated-one-hot' vector means a tensor which, for each slice along the
    batch dimension (assumed to be dimension 0), is 1.0 in the entry
    corresponding to the given index and 0 elsewhere.

    Note that B can be any matrix satisfying B * B^T = H where H is the Hessian,
    but will agree with the one used in the other methods of this class.

    Args:
      index: A tuple representing in the index of the entry in each slice that
        is 1.0. Note that len(index) must be equal to the number of elements
        of the 'hessian_factor_inner_shape' tensor minus one.

    Returns:
      The vector right-multiplied by B^T. Will be of the same shape(s) as the
      'inputs' property.
    """
    pass

  @abc.abstractproperty
  def hessian_factor_inner_shape(self):
    """The shape of the tensor returned by multiply_hessian_factor."""
    pass

  @abc.abstractproperty
  def hessian_factor_inner_static_shape(self):
    """Static version of hessian_factor_inner_shape."""
    pass


@six.add_metaclass(abc.ABCMeta)
class NegativeLogProbLoss(LossFunction):
  """Abstract base class for loss functions that are negative log probs."""

  def __init__(self, seed=None):
    self._default_seed = seed
    super(NegativeLogProbLoss, self).__init__()

  @property
  def inputs(self):
    return self.params

  @abc.abstractproperty
  def params(self):
    """Parameters to the underlying distribution."""
    pass

  @abc.abstractmethod
  def multiply_fisher(self, vector):
    """Right-multiply a vector by the Fisher.

    Args:
      vector: The vector to multiply.  Must be the same shape(s) as the
        'inputs' property.

    Returns:
      The vector right-multiplied by the Fisher.  Will be of the same shape(s)
      as the 'inputs' property.
    """
    pass

  @abc.abstractmethod
  def multiply_fisher_factor(self, vector):
    """Right-multiply a vector by a factor B of the Fisher.

    Here the 'Fisher' is the Fisher information matrix (i.e. expected outer-
    product of gradients) with respect to the parameters of the underlying
    probability distribtion (whose log-prob defines the loss). Typically this
    will be block-diagonal across different cases in the batch, since the
    distribution is usually (but not always) conditionally iid across different
    cases.

    Note that B can be any matrix satisfying B * B^T = F where F is the Fisher,
    but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply.  Must be of the shape given by the
        'fisher_factor_inner_shape' property.

    Returns:
      The vector right-multiplied by B. Will be of the same shape(s) as the
      'inputs' property.
    """
    pass

  @abc.abstractmethod
  def multiply_fisher_factor_transpose(self, vector):
    """Right-multiply a vector by the transpose of a factor B of the Fisher.

    Here the 'Fisher' is the Fisher information matrix (i.e. expected outer-
    product of gradients) with respect to the parameters of the underlying
    probability distribtion (whose log-prob defines the loss). Typically this
    will be block-diagonal across different cases in the batch, since the
    distribution is usually (but not always) conditionally iid across different
    cases.

    Note that B can be any matrix satisfying B * B^T = F where F is the Fisher,
    but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply.  Must be the same shape(s) as the
        'inputs' property.

    Returns:
      The vector right-multiplied by B^T.  Will be of the shape given by the
      'fisher_factor_inner_shape' property.
    """
    pass

  @abc.abstractmethod
  def multiply_fisher_factor_replicated_one_hot(self, index):
    """Right-multiply a replicated-one-hot vector by a factor B of the Fisher.

    Here the 'Fisher' is the Fisher information matrix (i.e. expected outer-
    product of gradients) with respect to the parameters of the underlying
    probability distribtion (whose log-prob defines the loss). Typically this
    will be block-diagonal across different cases in the batch, since the
    distribution is usually (but not always) conditionally iid across different
    cases.

    A 'replicated-one-hot' vector means a tensor which, for each slice along the
    batch dimension (assumed to be dimension 0), is 1.0 in the entry
    corresponding to the given index and 0 elsewhere.

    Note that B can be any matrix satisfying B * B^T = H where H is the Fisher,
    but will agree with the one used in the other methods of this class.

    Args:
      index: A tuple representing in the index of the entry in each slice that
        is 1.0. Note that len(index) must be equal to the number of elements
        of the 'fisher_factor_inner_shape' tensor minus one.

    Returns:
      The vector right-multiplied by B. Will be of the same shape(s) as the
      'inputs' property.
    """
    pass

  @abc.abstractproperty
  def fisher_factor_inner_shape(self):
    """The shape of the tensor returned by multiply_fisher_factor."""
    pass

  @abc.abstractproperty
  def fisher_factor_inner_static_shape(self):
    """Static version of fisher_factor_inner_shape."""
    pass

  @abc.abstractmethod
  def sample(self, seed):
    """Sample 'targets' from the underlying distribution."""
    pass

  def evaluate_on_sample(self, seed=None):
    """Evaluates the log probability on a random sample.

    Args:
      seed: int or None. Random seed for this draw from the distribution.

    Returns:
      Log probability of sampled targets, summed across examples.
    """
    if seed is None:
      seed = self._default_seed
    # We treat the targets as "constant".  It's only the inputs that get
    # "back-propped" through.
    return self._evaluate(array_ops.stop_gradient(self.sample(seed)))


# TODO(jamesmartens): should this just inherit from object to avoid "diamond"
# inheritance, or is there a better way?
class NaturalParamsNegativeLogProbLoss(NegativeLogProbLoss):
  """Base class for neg log prob losses whose inputs are 'natural' parameters.

  Note that the Hessian and Fisher for natural parameters of exponential-
  family models are the same, hence the purpose of this class.
  See here: https://arxiv.org/abs/1412.1193

  'Natural parameters' are defined for exponential-family models. See for
  example: https://en.wikipedia.org/wiki/Exponential_family
  """

  def multiply_hessian(self, vector):
    return self.multiply_fisher(vector)

  def multiply_hessian_factor(self, vector):
    return self.multiply_fisher_factor(vector)

  def multiply_hessian_factor_transpose(self, vector):
    return self.multiply_fisher_factor_transpose(vector)

  def multiply_hessian_factor_replicated_one_hot(self, index):
    return self.multiply_fisher_factor_replicated_one_hot(index)

  @property
  def hessian_factor_inner_shape(self):
    return self.fisher_factor_inner_shape

  @property
  def hessian_factor_inner_static_shape(self):
    return self.fisher_factor_inner_shape


class DistributionNegativeLogProbLoss(NegativeLogProbLoss):
  """Base class for neg log prob losses that use the TF Distribution classes."""

  def __init__(self, seed=None):
    super(DistributionNegativeLogProbLoss, self).__init__(seed=seed)

  @abc.abstractproperty
  def dist(self):
    """The underlying tf.distributions.Distribution."""
    pass

  def _evaluate(self, targets):
    return -math_ops.reduce_sum(self.dist.log_prob(targets))

  def sample(self, seed):
    return self.dist.sample(seed=seed)


class NormalMeanNegativeLogProbLoss(DistributionNegativeLogProbLoss,
                                    NaturalParamsNegativeLogProbLoss):
  """Neg log prob loss for a normal distribution parameterized by a mean vector.


  Note that the covariance is treated as a constant 'var' times the identity.
  Also note that the Fisher for such a normal distribution with respect the mean
  parameter is given by:

     F = (1/var) * I

  See for example https://www.ii.pwr.edu.pl/~tomczak/PDF/[JMT]Fisher_inf.pdf.
  """

  def __init__(self, mean, var=0.5, targets=None, seed=None):
    self._mean = mean
    self._var = var
    self._targets = targets
    super(NormalMeanNegativeLogProbLoss, self).__init__(seed=seed)

  @property
  def targets(self):
    return self._targets

  @property
  def dist(self):
    return normal.Normal(loc=self._mean, scale=math_ops.sqrt(self._var))

  @property
  def params(self):
    return self._mean

  def multiply_fisher(self, vector):
    return (1. / self._var) * vector

  def multiply_fisher_factor(self, vector):
    return self._var**-0.5 * vector

  def multiply_fisher_factor_transpose(self, vector):
    return self.multiply_fisher_factor(vector)  # it's symmetric in this case

  def multiply_fisher_factor_replicated_one_hot(self, index):
    assert len(index) == 1, "Length of index was {}".format(len(index))
    ones_slice = array_ops.expand_dims(
        array_ops.ones(array_ops.shape(self._mean)[:1], dtype=self._mean.dtype),
        axis=-1)
    output_slice = self._var**-0.5 * ones_slice
    return insert_slice_in_zeros(output_slice, 1, int(self._mean.shape[1]),
                                 index[0])

  @property
  def fisher_factor_inner_shape(self):
    return array_ops.shape(self._mean)

  @property
  def fisher_factor_inner_static_shape(self):
    return self._mean.shape


class NormalMeanVarianceNegativeLogProbLoss(DistributionNegativeLogProbLoss):
  """Negative log prob loss for a normal distribution with mean and variance.

  This class parameterizes a multivariate normal distribution with n independent
  dimensions. Unlike `NormalMeanNegativeLogProbLoss`, this class does not
  assume the variance is held constant. The Fisher Information for n = 1
  is given by,

  F = [[1 / variance,                0],
       [           0, 0.5 / variance^2]]

  where the parameters of the distribution are concatenated into a single
  vector as [mean, variance]. For n > 1, the mean parameter vector is
  concatenated with the variance parameter vector.

  See https://www.ii.pwr.edu.pl/~tomczak/PDF/[JMT]Fisher_inf.pdf for derivation.
  """

  def __init__(self, mean, variance, targets=None, seed=None):
    assert len(mean.shape) == 2, "Expect 2D mean tensor."
    assert len(variance.shape) == 2, "Expect 2D variance tensor."
    self._mean = mean
    self._variance = variance
    self._targets = targets
    super(NormalMeanVarianceNegativeLogProbLoss, self).__init__(seed=seed)

  @property
  def targets(self):
    return self._targets

  @property
  def dist(self):
    return normal.Normal(loc=self._mean, scale=math_ops.sqrt(self._variance))

  @property
  def params(self):
    return self._mean, self._variance

  def _concat(self, mean, variance):
    return array_ops.concat([mean, variance], axis=-1)

  def _split(self, params):
    return array_ops.split(params, 2, axis=-1)

  @property
  def _fisher_mean(self):
    return 1. / self._variance

  @property
  def _fisher_mean_factor(self):
    return 1. / math_ops.sqrt(self._variance)

  @property
  def _fisher_var(self):
    return 1. / (2 * math_ops.square(self._variance))

  @property
  def _fisher_var_factor(self):
    return 1. / (math_ops.sqrt(2.) * self._variance)

  def multiply_fisher(self, vecs):
    mean_vec, var_vec = vecs
    return (self._fisher_mean * mean_vec, self._fisher_var * var_vec)

  def multiply_fisher_factor(self, vecs):
    mean_vec, var_vec = self._split(vecs)
    return (self._fisher_mean_factor * mean_vec,
            self._fisher_var_factor * var_vec)

  def multiply_fisher_factor_transpose(self, vecs):
    mean_vec, var_vec = vecs
    return self._concat(self._fisher_mean_factor * mean_vec,
                        self._fisher_var_factor * var_vec)

  def multiply_fisher_factor_replicated_one_hot(self, index):
    assert len(index) == 1, "Length of index was {}".format(len(index))
    index = index[0]

    if index < int(self._mean.shape[-1]):
      # Index corresponds to mean parameter.
      mean_slice = self._fisher_mean_factor[:, index]
      mean_slice = array_ops.expand_dims(mean_slice, axis=-1)
      mean_output = insert_slice_in_zeros(mean_slice, 1, int(
          self._mean.shape[1]), index)
      var_output = array_ops.zeros_like(mean_output)
    else:
      index -= int(self._mean.shape[-1])
      # Index corresponds to variance parameter.
      var_slice = self._fisher_var_factor[:, index]
      var_slice = array_ops.expand_dims(var_slice, axis=-1)
      var_output = insert_slice_in_zeros(var_slice, 1,
                                         int(self._variance.shape[1]), index)
      mean_output = array_ops.zeros_like(var_output)

    return mean_output, var_output

  @property
  def fisher_factor_inner_shape(self):
    return array_ops.concat(
        [
            array_ops.shape(self._mean)[:-1],
            2 * array_ops.shape(self._mean)[-1:]
        ],
        axis=0)

  @property
  def fisher_factor_inner_static_shape(self):
    shape = self._mean.shape.as_list()
    return tensor_shape.TensorShape(shape[-1:] + [2 * shape[-1]])

  def multiply_hessian(self, vector):
    raise NotImplementedError()

  def multiply_hessian_factor(self, vector):
    raise NotImplementedError()

  def multiply_hessian_factor_transpose(self, vector):
    raise NotImplementedError()

  def multiply_hessian_factor_replicated_one_hot(self, index):
    raise NotImplementedError()

  @property
  def hessian_factor_inner_shape(self):
    raise NotImplementedError()

  @property
  def hessian_factor_inner_static_shape(self):
    raise NotImplementedError()


class CategoricalLogitsNegativeLogProbLoss(DistributionNegativeLogProbLoss,
                                           NaturalParamsNegativeLogProbLoss):
  """Neg log prob loss for a categorical distribution parameterized by logits.


  Note that the Fisher (for a single case) of a categorical distribution, with
  respect to the natural parameters (i.e. the logits), is given by:

  F = diag(p) - p*p^T

  where p = softmax(logits).  F can be factorized as F = B * B^T where

  B = diag(q) - p*q^T

  where q is the entry-wise square root of p. This is easy to verify using the
  fact that q^T*q = 1.
  """

  def __init__(self, logits, targets=None, seed=None):
    """Instantiates a CategoricalLogitsNegativeLogProbLoss.

    Args:
      logits: Tensor of shape [batch_size, output_size]. Parameters for
        underlying distribution.
      targets: None or Tensor of shape [output_size]. Each elements contains an
        index in [0, output_size).
      seed: int or None. Default random seed when sampling.
    """
    self._logits = logits
    self._targets = targets
    super(CategoricalLogitsNegativeLogProbLoss, self).__init__(seed=seed)

  @property
  def targets(self):
    return self._targets

  @property
  def dist(self):
    return categorical.Categorical(logits=self._logits)

  @property
  def _probs(self):
    return self.dist.probs

  @property
  def _sqrt_probs(self):
    return math_ops.sqrt(self._probs)

  @property
  def params(self):
    return self._logits

  def multiply_fisher(self, vector):
    probs = self._probs
    return vector * probs - probs * math_ops.reduce_sum(
        vector * probs, axis=-1, keep_dims=True)

  def multiply_fisher_factor(self, vector):
    probs = self._probs
    sqrt_probs = self._sqrt_probs
    return sqrt_probs * vector - probs * math_ops.reduce_sum(
        sqrt_probs * vector, axis=-1, keep_dims=True)

  def multiply_fisher_factor_transpose(self, vector):
    probs = self._probs
    sqrt_probs = self._sqrt_probs
    return sqrt_probs * vector - sqrt_probs * math_ops.reduce_sum(
        probs * vector, axis=-1, keep_dims=True)

  def multiply_fisher_factor_replicated_one_hot(self, index):
    assert len(index) == 1, "Length of index was {}".format(len(index))
    probs = self._probs
    sqrt_probs = self._sqrt_probs
    sqrt_probs_slice = array_ops.expand_dims(sqrt_probs[:, index[0]], -1)
    padded_slice = insert_slice_in_zeros(sqrt_probs_slice, 1,
                                         int(sqrt_probs.shape[1]), index[0])
    return padded_slice - probs * sqrt_probs_slice

  @property
  def fisher_factor_inner_shape(self):
    return array_ops.shape(self._logits)

  @property
  def fisher_factor_inner_static_shape(self):
    return self._logits.shape


class MultiBernoulliNegativeLogProbLoss(DistributionNegativeLogProbLoss,
                                        NaturalParamsNegativeLogProbLoss):
  """Neg log prob loss for multiple Bernoulli distributions param'd by logits.

  Represents N independent Bernoulli distributions where N = len(logits). Its
  Fisher Information matrix is given by,

  F = diag(p * (1-p))
  p = sigmoid(logits)

  As F is diagonal with positive entries, its factor B is,

  B = diag(sqrt(p * (1-p)))
  """

  def __init__(self, logits, targets=None, seed=None):
    self._logits = logits
    self._targets = targets
    super(MultiBernoulliNegativeLogProbLoss, self).__init__(seed=seed)

  @property
  def targets(self):
    return self._targets

  @property
  def dist(self):
    return bernoulli.Bernoulli(logits=self._logits)

  @property
  def _probs(self):
    return self.dist.probs

  @property
  def params(self):
    return self._logits

  def multiply_fisher(self, vector):
    return self._probs * (1 - self._probs) * vector

  def multiply_fisher_factor(self, vector):
    return math_ops.sqrt(self._probs * (1 - self._probs)) * vector

  def multiply_fisher_factor_transpose(self, vector):
    return self.multiply_fisher_factor(vector)  # it's symmetric in this case

  def multiply_fisher_factor_replicated_one_hot(self, index):
    assert len(index) == 1, "Length of index was {}".format(len(index))
    probs_slice = array_ops.expand_dims(self._probs[:, index[0]], -1)
    output_slice = math_ops.sqrt(probs_slice * (1 - probs_slice))
    return insert_slice_in_zeros(output_slice, 1, int(self._logits.shape[1]),
                                 index[0])

  @property
  def fisher_factor_inner_shape(self):
    return array_ops.shape(self._logits)

  @property
  def fisher_factor_inner_static_shape(self):
    return self._logits.shape


def insert_slice_in_zeros(slice_to_insert, dim, dim_size, position):
  """Inserts slice into a larger tensor of zeros.

  Forms a new tensor which is the same shape as slice_to_insert, except that
  the dimension given by 'dim' is expanded to the size given by 'dim_size'.
  'position' determines the position (index) at which to insert the slice within
  that dimension.

  Assumes slice_to_insert.shape[dim] = 1.

  Args:
    slice_to_insert: The slice to insert.
    dim: The dimension which to expand with zeros.
    dim_size: The new size of the 'dim' dimension.
    position: The position of 'slice_to_insert' in the new tensor.

  Returns:
    The new tensor.

  Raises:
    ValueError: If the slice's shape at the given dim is not 1.
  """
  slice_shape = slice_to_insert.shape
  if slice_shape[dim] != 1:
    raise ValueError("Expected slice_to_insert.shape to have {} dim of 1, but "
                     "was {}".format(dim, slice_to_insert.shape[dim]))

  before = [0] * int(len(slice_shape))
  after = before[:]
  before[dim] = position
  after[dim] = dim_size - position - 1

  return array_ops.pad(slice_to_insert, list(zip(before, after)))


class OnehotCategoricalLogitsNegativeLogProbLoss(
    CategoricalLogitsNegativeLogProbLoss):
  """Neg log prob loss for a categorical distribution with onehot targets.

  Identical to CategoricalLogitsNegativeLogProbLoss except that the underlying
  distribution is OneHotCategorical as opposed to Categorical.
  """

  @property
  def dist(self):
    return onehot_categorical.OneHotCategorical(logits=self._logits)
