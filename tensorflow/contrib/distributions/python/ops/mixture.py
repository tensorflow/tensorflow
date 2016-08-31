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
"""The Mixture distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import categorical
from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


class Mixture(distribution.Distribution):
  """Mixture distribution.

  The `Mixture` object implements batched mixture distributions.
  The mixture model is defined by a `Categorical` distribution (the mixture)
  and a python list of `Distribution` objects.

  Methods supported include `log_prob`, `prob`, `mean`, `sample`, and
  `entropy_lower_bound`.
  """

  def __init__(self,
               cat,
               components,
               validate_args=True,
               allow_nan_stats=False,
               name="Mixture"):
    """Initialize a Mixture distribution.

    A `Mixture` is defined by a `Categorical` (`cat`, representing the
    mixture probabilities) and a list of `Distribution` objects
    all having matching dtype, batch shape, event shape, and continuity
    properties (the components).

    The user does not pass the list of distributions directly, but rather a
    list of `(constructor, batch_tensor_params_dict)` pairs,
    called `components`. The list of distributions is created via:

    ```python
    distributions = [
      c(**params_dict) for (c, params_dict) in zip(*components)
    ]
    ```

    This form allows for certain types of batch-shape optimizations within
    this class.

    An example of `components`:

    ```python
    components = [
      (tf.contrib.distributions.Normal, {"mu": 3.0, "sigma": 1.0}),
      (functools.partial(tf.contrib.distributions.Normal, validate_args=False),
       {"mu": 3.0, "sigma": 2.0}),
      (tf.contrib.distributions.Normal.from_params,
       {"mu": 1.0, "sigma": -1.0})
    ]
    ```

    The `num_classes` of `cat` must be possible to infer at graph construction
    time and match `len(distributions)`.

    Args:
      cat: A `Categorical` distribution instance, representing the probabilities
          of `distributions`.
      components: A list or tuple of `(constructor, batch_tensor_params)`
        tuples.  The `constructor` must be a callable, and `batch_tensor_params`
        must be a dict mapping constructor kwargs to batchwise parameters.
        Each `Distribution` instance created by calling
        `constructor(**batch_tensor_params)` must have the same type, be defined
        on the same domain, and have matching `event_shape` and `batch_shape`.
      validate_args: Boolean, default `True`.  If `True`, raise a runtime error
        if batch or event ranks are inconsistent between cat and any of the
        distributions.  This is only checked if the ranks cannot be determined
        statically at graph construction time.
      allow_nan_stats: Boolean, default `False`.  If `False`, raise an
       exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: A name for this distribution (optional).

    Raises:
      TypeError: If cat is not a `Categorical`, or `components` is not
        a list or tuple, or the elements of `components` are not
        tuples of the form `(callable, dict)`, or the objects resulting
        from calling `callable(**dict)` are not instances of `Distribution`, or
        the resulting instances of `Distribution` do not have matching
        continuity properties, or do not have matching `dtype`.
      ValueError: If `components` is an empty list or tuple, or the
        distributions created from `components` do have a statically known event
        rank.  If `cat.num_classes` cannot be inferred at graph creation time,
        or the constant value of `cat.num_classes` is not equal to
        `len(distributions)`, or all `distributions` and `cat` do not have
        matching static batch shapes, or all components' distributions do not
        have matching static event shapes.
    """
    if not isinstance(cat, categorical.Categorical):
      raise TypeError("cat must be a Categorical distribution, but saw: %s" %
                      cat)
    if not components:
      raise ValueError("components must be a non-empty list or tuple")
    if not isinstance(components, (list, tuple)):
      raise TypeError("components must be a list or tuple, but saw: %s" %
                      components)
    if not all(isinstance(c, tuple) and len(c) == 2 and
               callable(c[0]) and isinstance(c[1], dict)
               for c in components):
      raise TypeError(
          "all entries in components must be tuples of the form "
          "(make, params), where make is callable and params is a dict,"
          " but saw: %s" % components)

    def _make_tensors(d):
      return dict((k, ops.convert_to_tensor(v, name="tensor_%s" % k))
                  for (k, v) in d.items())

    with ops.name_scope(name, values=[cat.logits]):
      components_tensor_params = list((make, _make_tensors(batch_params))
                                      for (make, batch_params) in components)
      distributions = [make(**batch_params)
                       for (make, batch_params) in components_tensor_params]

    # Store components internally with their batch params having been
    # converted to tensors.
    # TODO(ebrevdo): Use self._components to optimize sampling.
    self._components = components_tensor_params

    if not all(isinstance(d, distribution.Distribution) for d in distributions):
      raise TypeError(
          "all entries in distributions must be instances of Distribution, "
          "but saw: %s" % distributions)

    dtype = distributions[0].dtype
    if not all(d.dtype == dtype for d in distributions):
      raise TypeError("All distributions must have the same dtype, but saw "
                      "dtypes: %s" % [(d.name, d.dtype) for d in distributions])
    is_continuous = distributions[0].is_continuous
    if not all(d.is_continuous == is_continuous for d in distributions):
      raise TypeError(
          "All distributions must either be continuous or not, but continuity "
          "values are: %s" % [(d.name, d.is_continuous) for d in distributions])
    static_event_shape = distributions[0].get_event_shape()
    static_batch_shape = cat.get_batch_shape()
    for d in distributions:
      static_event_shape = static_event_shape.merge_with(d.get_event_shape())
      static_batch_shape = static_batch_shape.merge_with(d.get_batch_shape())
    if static_event_shape.ndims is None:
      raise ValueError(
          "Expected to know rank(event_shape) from distributions, but "
          "none of the distributions provide a static number of ndims")

    # Ensure that all batch and event ndims are consistent.
    with ops.name_scope(name, values=[cat.logits]):
      num_components = cat.num_classes
      static_num_components = tensor_util.constant_value(num_components)
      if static_num_components is None:
        raise ValueError(
            "Could not infer number of classes from cat and unable "
            "to compare this value to the number of distributions passed in.")
      # Possibly convert from numpy 0-D array.
      static_num_components = int(static_num_components)
      if static_num_components != len(distributions):
        raise ValueError("cat.num_classes != len(distributions): %d vs. %d" %
                         (static_num_components, len(distributions)))

      cat_batch_shape = cat.batch_shape()
      cat_batch_rank = array_ops.size(cat_batch_shape)
      if validate_args:
        batch_shapes = [d.batch_shape() for d in distributions]
        batch_ranks = [array_ops.size(bs) for bs in batch_shapes]
        check_message = ("distributions[%d] batch shape must match cat "
                         "batch shape")
        self._assertions = [
            check_ops.assert_equal(
                cat_batch_rank, batch_ranks[di], message=check_message % di)
            for di in range(len(distributions))
        ]
        self._assertions += [
            check_ops.assert_equal(
                cat_batch_shape, batch_shapes[di], message=check_message % di)
            for di in range(len(distributions))
        ]
      else:
        self._assertions = []

      self._cat = cat
      self._distributions = list(distributions)
      self._num_components = static_num_components
      self._static_event_shape = static_event_shape
      self._static_batch_shape = static_batch_shape

      super(Mixture, self).__init__(
          dtype=dtype,
          parameters={"cat": self._cat, "distributions": self._distributions,
                      "num_components": self._num_components},
          is_reparameterized=False,
          is_continuous=is_continuous,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)

  @property
  def cat(self):
    return self._cat

  @property
  def distributions(self):
    return self._distributions

  @property
  def num_components(self):
    return self._num_components

  def _batch_shape(self):
    return self._cat.batch_shape()

  def _get_batch_shape(self):
    return self._static_batch_shape

  def _event_shape(self):
    return self._distributions[0].event_shape()

  def _get_event_shape(self):
    return self._static_event_shape

  def _mean(self):
    with ops.control_dependencies(self._assertions):
      distribution_means = [d.mean() for d in self.distributions]
      cat_probs = self._cat_probs(log_probs=False)
      # This was checked to not be None at construction time.
      static_event_rank = self.get_event_shape().ndims
      # Expand the rank of x up to static_event_rank times so that
      # broadcasting works correctly.
      def expand(x):
        expanded_x = x
        for _ in range(static_event_rank):
          expanded_x = array_ops.expand_dims(expanded_x, -1)
        return expanded_x
      cat_probs = [expand(c_p) for c_p in cat_probs]
      partial_means = [
          c_p * m for (c_p, m) in zip(cat_probs, distribution_means)
      ]
      # These should all be the same shape by virtue of matching
      # batch_shape and event_shape.
      return math_ops.add_n(partial_means)

  def _log_prob(self, x):
    with ops.control_dependencies(self._assertions):
      x = ops.convert_to_tensor(x, name="x")
      distribution_log_probs = [d.log_prob(x) for d in self.distributions]
      cat_log_probs = self._cat_probs(log_probs=True)
      final_log_probs = [
          cat_lp + d_lp
          for (cat_lp, d_lp) in zip(cat_log_probs, distribution_log_probs)
      ]
      concat_log_probs = array_ops.pack(final_log_probs, 0)
      log_sum_exp = math_ops.reduce_logsumexp(concat_log_probs, [0])
      return log_sum_exp

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _sample_n(self, n, seed=None):
    with ops.control_dependencies(self._assertions):
      n = ops.convert_to_tensor(n, name="n")
      static_n = tensor_util.constant_value(n)
      n = int(static_n) if static_n is not None else n
      cat_samples = self.cat.sample_n(n, seed=seed)

      static_samples_shape = cat_samples.get_shape()
      if static_samples_shape.is_fully_defined():
        samples_shape = static_samples_shape.as_list()
        samples_size = static_samples_shape.num_elements()
      else:
        samples_shape = array_ops.shape(cat_samples)
        samples_size = array_ops.size(cat_samples)
      static_batch_shape = self.get_batch_shape()
      if static_batch_shape.is_fully_defined():
        batch_shape = static_batch_shape.as_list()
        batch_size = static_batch_shape.num_elements()
      else:
        batch_shape = self.batch_shape()
        batch_size = array_ops.reduce_prod(batch_shape)
      static_event_shape = self.get_event_shape()
      if static_event_shape.is_fully_defined():
        event_shape = np.array(static_event_shape.as_list(), dtype=np.int32)
      else:
        event_shape = self.event_shape()

      # Get indices into the raw cat sampling tensor.  We will
      # need these to stitch sample values back out after sampling
      # within the component partitions.
      samples_raw_indices = array_ops.reshape(
          math_ops.range(0, samples_size), samples_shape)

      # Partition the raw indices so that we can use
      # dynamic_stitch later to reconstruct the samples from the
      # known partitions.
      partitioned_samples_indices = data_flow_ops.dynamic_partition(
          data=samples_raw_indices,
          partitions=cat_samples,
          num_partitions=self.num_components)

      # Copy the batch indices n times, as we will need to know
      # these to pull out the appropriate rows within the
      # component partitions.
      batch_raw_indices = array_ops.reshape(
          array_ops.tile(math_ops.range(0, batch_size), [n]), samples_shape)

      # Explanation of the dynamic partitioning below:
      #   batch indices are i.e., [0, 1, 0, 1, 0, 1]
      # Suppose partitions are:
      #     [1 1 0 0 1 1]
      # After partitioning, batch indices are cut as:
      #     [batch_indices[x] for x in 2, 3]
      #     [batch_indices[x] for x in 0, 1, 4, 5]
      # i.e.
      #     [1 1] and [0 0 0 0]
      # Now we sample n=2 from part 0 and n=4 from part 1.
      # For part 0 we want samples from batch entries 1, 1 (samples 0, 1),
      # and for part 1 we want samples from batch entries 0, 0, 0, 0
      #   (samples 0, 1, 2, 3).
      partitioned_batch_indices = data_flow_ops.dynamic_partition(
          data=batch_raw_indices,
          partitions=cat_samples,
          num_partitions=self.num_components)
      samples_class = [None for _ in range(self.num_components)]
      for c in range(self.num_components):
        n_class = array_ops.size(partitioned_samples_indices[c])
        samples_class_c = self.distributions[c].sample_n(n_class, seed=seed)

        # Pull out the correct batch entries from each index.
        # To do this, we may have to flatten the batch shape.

        # For sample s, batch element b of component c, we get the
        # partitioned batch indices from
        # partitioned_batch_indices[c]; and shift each element by
        # the sample index.  The final lookup can be thought of as
        # a matrix gather along locations (s, b) in
        # samples_class_c where the n_class rows correspond to
        # samples within this component and the batch_size columns
        # correspond to batch elements within the component.
        #
        # Thus the lookup index is
        #   lookup[c, i] = batch_size * s[i] + b[c, i]
        # for i = 0 ... n_class[c] - 1.
        lookup_partitioned_batch_indices = (
            batch_size * math_ops.range(n_class) +
            partitioned_batch_indices[c])

        # Try to avoid a reshape to make the sample + batch one
        # row (for array_ops.gather).  This can be done only when
        # the batch shape is known and is rank 1.
        if static_batch_shape.ndims == 1:
          samples_class_c = array_ops.gather(
              samples_class_c, lookup_partitioned_batch_indices)
        else:
          samples_class_c = array_ops.reshape(
              samples_class_c,
              array_ops.concat(0, ([n_class * batch_size], event_shape)))
          samples_class_c = array_ops.gather(
              samples_class_c, lookup_partitioned_batch_indices)
        samples_class[c] = samples_class_c

      # Stitch back together the samples across the components.
      lhs_flat_ret = data_flow_ops.dynamic_stitch(
          indices=partitioned_samples_indices, data=samples_class)
      # Reshape back to proper sample, batch, and event shape.
      ret = array_ops.reshape(lhs_flat_ret,
                              array_ops.concat(0, (samples_shape,
                                                   self.event_shape())))
      ret.set_shape(
          tensor_shape.TensorShape(static_samples_shape).concatenate(
              self.get_event_shape()))
      return ret

  def entropy_lower_bound(self, name="entropy_lower_bound"):
    r"""A lower bound on the entropy of this mixture model.

    The bound below is not always very tight, and its usefulness depends
    on the mixture probabilities and the distributions in use.

    A lower bound is useful for ELBO when the `Mixture` is the variational
    distribution:

    \\(
    \log p(x) >= ELBO = \int q(z) \log p(x, z) dz + H[q]
    \\)

    where \\( p \\) is the prior disribution, \\( q \\) is the variational,
    and \\( H[q] \\) is the entropy of \\( q \\).  If there is a lower bound
    \\( G[q] \\) such that \\( H[q] \geq G[q] \\) then it can be used in
    place of \\( H[q] \\).

    For a mixture of distributions \\( q(Z) = \sum_i c_i q_i(Z) \\) with
    \\( \sum_i c_i = 1 \\), by the concavity of \\( f(x) = -x \log x \\), a
    simple lower bound is:

    \\(
    \begin{align}
    H[q] & = - \int q(z) \log q(z) dz \\\
       & = - \int (\sum_i c_i q_i(z)) \log(\sum_i c_i q_i(z)) dz \\\
       & \geq - \sum_i c_i \int q_i(z) \log q_i(z) dz \\\
       & = \sum_i c_i H[q_i]
    \end{align}
    \\)

    This is the term we calculate below for \\( G[q] \\).

    Args:
      name: A name for this operation (optional).

    Returns:
      A lower bound on the Mixture's entropy.
    """
    with self._name_scope(name, values=[self.cat.logits]):
      with ops.control_dependencies(self._assertions):
        distribution_entropies = [d.entropy() for d in self.distributions]
        cat_probs = self._cat_probs(log_probs=False)
        partial_entropies = [
            c_p * m for (c_p, m) in zip(cat_probs, distribution_entropies)
        ]
        # These are all the same shape by virtue of matching batch_shape
        return math_ops.add_n(partial_entropies)

  def _cat_probs(self, log_probs):
    """Get a list of num_components batchwise probabilities."""
    which_softmax = nn_ops.log_softmax if log_probs else nn_ops.softmax
    cat_probs = which_softmax(self.cat.logits)
    cat_probs = array_ops.unpack(
        cat_probs, num=self.num_components, axis=-1)
    return cat_probs
