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

"""TensorFlow interface for third-party optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging


__all__ = ['ExternalOptimizerInterface', 'ScipyOptimizerInterface']


class ExternalOptimizerInterface(object):
  """Base class for interfaces with external optimization algorithms.

  Subclass this and implement `_minimize` in order to wrap a new optimization
  algorithm.

  `ExternalOptimizerInterface` should not be instantiated directly; instead use
  e.g. `ScipyOptimizerInterface`.

  @@__init__

  @@minimize
  """

  def __init__(self, loss, var_list=None, equalities=None, inequalities=None,
               **optimizer_kwargs):
    """Initialize a new interface instance.

    Args:
      loss: A scalar `Tensor` to be minimized.
      var_list: Optional list of `Variable` objects to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      equalities: Optional list of equality constraint scalar `Tensor`s to be
        held equal to zero.
      inequalities: Optional list of inequality constraint scalar `Tensor`s
        to be kept nonnegative.
      **optimizer_kwargs: Other subclass-specific keyword arguments.
    """
    self._loss = loss
    self._equalities = equalities or []
    self._inequalities = inequalities or []

    if var_list is None:
      self._vars = variables.trainable_variables()
    else:
      self._vars = list(var_list)

    self._update_placeholders = [array_ops.placeholder(var.dtype)
                                 for var in self._vars]
    self._var_updates = [var.assign(array_ops.reshape(placeholder,
                                                      _get_shape_tuple(var)))
                         for var, placeholder in
                         zip(self._vars, self._update_placeholders)]

    loss_grads = _compute_gradients(loss, self._vars)
    equalities_grads = [_compute_gradients(equality, self._vars)
                        for equality in self._equalities]
    inequalities_grads = [_compute_gradients(inequality, self._vars)
                          for inequality in self._inequalities]

    self.optimizer_kwargs = optimizer_kwargs

    self._packed_var = self._pack(self._vars)
    self._packed_loss_grad = self._pack(loss_grads)
    self._packed_equality_grads = [
        self._pack(equality_grads)
        for equality_grads in equalities_grads
    ]
    self._packed_inequality_grads = [
        self._pack(inequality_grads)
        for inequality_grads in inequalities_grads
    ]

    dims = [_prod(_get_shape_tuple(var)) for var in self._vars]
    accumulated_dims = list(_accumulate(dims))
    self._packing_slices = [
        slice(start, end) for start, end in zip(accumulated_dims[:-1],
                                                accumulated_dims[1:])]

  def minimize(self, session=None, feed_dict=None, fetches=None,
               step_callback=None, loss_callback=None):
    """Minimize a scalar `Tensor`.

    Variables subject to optimization are updated in-place at the end of
    optimization.

    Note that this method does *not* just return a minimization `Op`, unlike
    `Optimizer.minimize()`; instead it actually performs minimization by
    executing commands to control a `Session`.

    Args:
      session: A `Session` instance.
      feed_dict: A feed dict to be passed to calls to `session.run`.
      fetches: A list of `Tensor`s to fetch and supply to `loss_callback`
        as positional arguments.
      step_callback: A function to be called at each optimization step;
        arguments are the current values of all optimization variables
        flattened into a single vector.
      loss_callback: A function to be called every time the loss and gradients
        are computed, with evaluated fetches supplied as positional arguments.
    """
    session = session or ops.get_default_session()
    feed_dict = feed_dict or {}
    fetches = fetches or []

    loss_callback = loss_callback or (lambda *fetches: None)
    step_callback = step_callback or (lambda xk: None)

    # Construct loss function and associated gradient.
    loss_grad_func = self._make_eval_func(
        [self._loss, self._packed_loss_grad],
        session, feed_dict, fetches, loss_callback)

    # Construct equality constraint functions and associated gradients.
    equality_funcs = self._make_eval_funcs(
        self._equalities, session, feed_dict, fetches)
    equality_grad_funcs = self._make_eval_funcs(
        self._packed_equality_grads, session, feed_dict, fetches)

    # Construct inequality constraint functions and associated gradients.
    inequality_funcs = self._make_eval_funcs(
        self._inequalities, session, feed_dict, fetches)
    inequality_grad_funcs = self._make_eval_funcs(
        self._packed_inequality_grads, session, feed_dict, fetches)

    # Get initial value from TF session.
    initial_packed_var_val = session.run(self._packed_var)

    # Perform minimization.
    packed_var_val = self._minimize(
        initial_val=initial_packed_var_val, loss_grad_func=loss_grad_func,
        equality_funcs=equality_funcs,
        equality_grad_funcs=equality_grad_funcs,
        inequality_funcs=inequality_funcs,
        inequality_grad_funcs=inequality_grad_funcs,
        step_callback=step_callback, optimizer_kwargs=self.optimizer_kwargs)
    var_vals = [packed_var_val[packing_slice]
                for packing_slice in self._packing_slices]

    # Set optimization variables to their new values.
    session.run(self._var_updates,
                feed_dict=dict(zip(self._update_placeholders, var_vals)))

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    """Wrapper for a particular optimization algorithm implementation.

    It would be appropriate for a subclass implementation of this method to
    raise `NotImplementedError` if unsupported arguments are passed: e.g. if an
    algorithm does not support constraints but `len(equality_funcs) > 0`.

    Args:
      initial_val: A NumPy vector of initial values.
      loss_grad_func: A function accepting a NumPy packed variable vector and
        returning two outputs, a loss value and the gradient of that loss with
        respect to the packed variable vector.
      equality_funcs: A list of functions each of which specifies a scalar
        quantity that an optimizer should hold exactly zero.
      equality_grad_funcs: A list of gradients of equality_funcs.
      inequality_funcs: A list of functions each of which specifies a scalar
        quantity that an optimizer should hold >= 0.
      inequality_grad_funcs: A list of gradients of inequality_funcs.
      step_callback: A callback function to execute at each optimization step,
        supplied with the current value of the packed variable vector.
      optimizer_kwargs: Other key-value arguments available to the optimizer.

    Returns:
      The optimal variable vector as a NumPy vector.
    """
    raise NotImplementedError(
        'To use ExternalOptimizerInterface, subclass from it and implement '
        'the _minimize() method.')

  @classmethod
  def _pack(cls, tensors):
    """Pack a list of `Tensor`s into a single, flattened, rank-1 `Tensor`."""
    if not tensors:
      return None
    elif len(tensors) == 1:
      return array_ops.reshape(tensors[0], [-1])
    else:
      flattened = [array_ops.reshape(tensor, [-1]) for tensor in tensors]
      return array_ops.concat(flattened, 0)

  def _make_eval_func(self, tensors, session, feed_dict, fetches,
                      callback=None):
    """Construct a function that evaluates a `Tensor` or list of `Tensor`s."""
    if not isinstance(tensors, list):
      tensors = [tensors]
    num_tensors = len(tensors)

    def eval_func(x):
      """Function to evaluate a `Tensor`."""
      augmented_feed_dict = {
          var: x[packing_slice].reshape(_get_shape_tuple(var))
          for var, packing_slice in zip(self._vars, self._packing_slices)
      }
      augmented_feed_dict.update(feed_dict)
      augmented_fetches = tensors + fetches

      augmented_fetch_vals = session.run(
          augmented_fetches, feed_dict=augmented_feed_dict)

      if callable(callback):
        callback(*augmented_fetch_vals[num_tensors:])

      return augmented_fetch_vals[:num_tensors]

    return eval_func

  def _make_eval_funcs(self, tensors, session, feed_dict, fetches,
                       callback=None):
    return [
        self._make_eval_func(tensor, session, feed_dict, fetches, callback)
        for tensor in tensors
    ]


class ScipyOptimizerInterface(ExternalOptimizerInterface):
  """Wrapper allowing `scipy.optimize.minimize` to operate a `tf.Session`.

  Example:

  ```python
  vector = tf.Variable([7., 7.], 'vector')

  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))

  optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})

  with tf.Session() as session:
    optimizer.minimize(session)

  # The value of vector should now be [0., 0.].
  ```

  Example with constraints:

  ```python
  vector = tf.Variable([7., 7.], 'vector')

  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
  # Ensure the vector's y component is = 1.
  equalities = [vector[1] - 1.]
  # Ensure the vector's x component is >= 1.
  inequalities = [vector[0] - 1.]

  # Our default SciPy optimization algorithm, L-BFGS-B, does not support
  # general constraints. Thus we use SLSQP instead.
  optimizer = ScipyOptimizerInterface(
      loss, equalities=equalities, inequalities=inequalities, method='SLSQP')

  with tf.Session() as session:
    optimizer.minimize(session)

  # The value of vector should now be [1., 1.].
  ```
  """

  _DEFAULT_METHOD = 'L-BFGS-B'

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    def loss_grad_func_wrapper(x):
      # SciPy's L-BFGS-B Fortran implementation requires gradients as doubles.
      loss, gradient = loss_grad_func(x)
      return loss, gradient.astype('float64')

    method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)

    constraints = []
    for func, grad_func in zip(equality_funcs, equality_grad_funcs):
      constraints.append({'type': 'eq', 'fun': func, 'jac': grad_func})
    for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
      constraints.append({'type': 'ineq', 'fun': func, 'jac': grad_func})

    minimize_args = [loss_grad_func_wrapper, initial_val]
    minimize_kwargs = {
        'jac': True,
        'callback': step_callback,
        'method': method,
        'constraints': constraints,
    }
    minimize_kwargs.update(optimizer_kwargs)
    if method == 'SLSQP':
      # SLSQP doesn't support step callbacks. Obviate associated warning
      # message.
      del minimize_kwargs['callback']

    import scipy.optimize  # pylint: disable=g-import-not-at-top
    result = scipy.optimize.minimize(*minimize_args, **minimize_kwargs)
    logging.info('Optimization terminated with:\n'
                 '  Message: %s\n'
                 '  Objective function value: %f\n'
                 '  Number of iterations: %d\n'
                 '  Number of functions evaluations: %d',
                 result.message, result.fun, result.nit, result.nfev)

    return result['x']


def _accumulate(list_):
  total = 0
  yield total
  for x in list_:
    total += x
    yield total


def _get_shape_tuple(tensor):
  return tuple(dim.value for dim in tensor.get_shape())


def _prod(array):
  prod = 1
  for value in array:
    prod *= value
  return prod


def _compute_gradients(tensor, var_list):
  grads = gradients.gradients(tensor, var_list)
  # tf.gradients sometimes returns `None` when it should return 0.
  return [grad if grad is not None else array_ops.zeros_like(var)
          for var, grad in zip(var_list, grads)]
