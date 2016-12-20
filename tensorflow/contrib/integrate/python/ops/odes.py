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

"""ODE solvers for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops


_ButcherTableau = collections.namedtuple(
    '_ButcherTableau', 'alpha beta c_sol c_mid c_error')

# Parameters from Shampine (1986), section 4.
_DORMAND_PRINCE_TABLEAU = _ButcherTableau(
    alpha=[1/5, 3/10, 4/5, 8/9, 1., 1.],
    beta=[[1/5],
          [3/40, 9/40],
          [44/45, -56/15, 32/9],
          [19372/6561, -25360/2187, 64448/6561, -212/729],
          [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
          [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]],
    c_sol=[35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
    c_mid=[6025192743/30085553152 / 2, 0, 51252292925/65400821598 / 2,
           -2691868925/45128329728 / 2, 187940372067/1594534317056 / 2,
           -1776094331/19743644256 / 2, 11237099/235043384 / 2],
    c_error=[1951/21600 - 35/384,
             0,
             22642/50085 - 500/1113,
             451/720 - 125/192,
             -12231/42400 - -2187/6784,
             649/6300 - 11/84,
             1/60],
)


def _possibly_nonzero(x):
  return isinstance(x, ops.Tensor) or x != 0


def _scaled_dot_product(scale, xs, ys, name=None):
  """Calculate a scaled, vector inner product between lists of Tensors."""
  with ops.name_scope(name, 'scaled_dot_product', [scale, xs, ys]) as scope:
    # Some of the parameters in our Butcher tableau include zeros. Using
    # _possibly_nonzero lets us avoid wasted computation.
    return math_ops.add_n([(scale * x) * y for x, y in zip(xs, ys)
                           if _possibly_nonzero(x) or _possibly_nonzero(y)],
                          name=scope)


def _dot_product(xs, ys, name=None):
  """Calculate the vector inner product between two lists of Tensors."""
  with ops.name_scope(name, 'dot_product', [xs, ys]) as scope:
    return math_ops.add_n([x * y for x, y in zip(xs, ys)], name=scope)


def _runge_kutta_step(func, y0, f0, t0, dt, tableau=_DORMAND_PRINCE_TABLEAU,
                      name=None):
  """Take an arbitrary Runge-Kutta step and estimate error.

  Args:
    func: Function to evaluate like `func(y, t)` to compute the time derivative
      of `y`.
    y0: Tensor initial value for the state.
    f0: Tensor initial value for the derivative, computed from `func(y0, t0)`.
    t0: float64 scalar Tensor giving the initial time.
    dt: float64 scalar Tensor giving the size of the desired time step.
    tableau: optional _ButcherTableau describing how to take the Runge-Kutta
      step.
    name: optional name for the operation.

  Returns:
    Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
    the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
    estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
    calculating these terms.
  """
  with ops.name_scope(name, 'runge_kutta_step', [y0, f0, t0, dt]) as scope:
    y0 = ops.convert_to_tensor(y0, name='y0')
    f0 = ops.convert_to_tensor(f0, name='f0')
    t0 = ops.convert_to_tensor(t0, name='t0')
    dt = ops.convert_to_tensor(dt, name='dt')
    dt_cast = math_ops.cast(dt, y0.dtype)

    k = [f0]
    for alpha_i, beta_i in zip(tableau.alpha, tableau.beta):
      ti = t0 + alpha_i * dt
      yi = y0 + _scaled_dot_product(dt_cast, beta_i, k)
      k.append(func(yi, ti))

    if not (tableau.c_sol[-1] == 0 and tableau.c_sol == tableau.beta[-1]):
      # This property (true for Dormand-Prince) lets us save a few FLOPs.
      yi = y0 + _scaled_dot_product(dt_cast, tableau.c_sol, k)

    y1 = array_ops.identity(yi, name='%s/y1' % scope)
    f1 = array_ops.identity(k[-1], name='%s/f1' % scope)
    y1_error = _scaled_dot_product(dt_cast, tableau.c_error, k,
                                   name='%s/y1_error' % scope)
    return (y1, f1, y1_error, k)


def _interp_fit(y0, y1, y_mid, f0, f1, dt):
  """Fit coefficients for 4th order polynomial interpolation.

  Args:
    y0: function value at the start of the interval.
    y1: function value at the end of the interval.
    y_mid: function value at the mid-point of the interval.
    f0: derivative value at the start of the interval.
    f1: derivative value at the end of the interval.
    dt: width of the interval.

  Returns:
    List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
    `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
    between 0 (start of interval) and 1 (end of interval).
  """
  # a, b, c, d, e = sympy.symbols('a b c d e')
  # x, dt, y0, y1, y_mid, f0, f1 = sympy.symbols('x dt y0 y1 y_mid f0 f1')
  # p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
  # sympy.solve([p.subs(x, 0) - y0,
  #              p.subs(x, 1 / 2) - y_mid,
  #              p.subs(x, 1) - y1,
  #              (p.diff(x) / dt).subs(x, 0) - f0,
  #              (p.diff(x) / dt).subs(x, 1) - f1],
  #             [a, b, c, d, e])
  # {a: -2.0*dt*f0 + 2.0*dt*f1 - 8.0*y0 - 8.0*y1 + 16.0*y_mid,
  #  b: 5.0*dt*f0 - 3.0*dt*f1 + 18.0*y0 + 14.0*y1 - 32.0*y_mid,
  #  c: -4.0*dt*f0 + dt*f1 - 11.0*y0 - 5.0*y1 + 16.0*y_mid,
  #  d: dt*f0,
  #  e: y0}
  a = _dot_product([-2 * dt, 2 * dt, -8, -8, 16], [f0, f1, y0, y1, y_mid])
  b = _dot_product([5 * dt, -3 * dt, 18, 14, -32], [f0, f1, y0, y1, y_mid])
  c = _dot_product([-4 * dt, dt, -11, -5, 16], [f0, f1, y0, y1, y_mid])
  d = dt * f0
  e = y0
  return [a, b, c, d, e]


def _interp_fit_rk(y0, y1, k, dt, tableau=_DORMAND_PRINCE_TABLEAU):
  """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
  with ops.name_scope('interp_fit_rk'):
    dt = math_ops.cast(dt, y0.dtype)
    y_mid = y0 + _scaled_dot_product(dt, tableau.c_mid, k)
    f0 = k[0]
    f1 = k[-1]
    return _interp_fit(y0, y1, y_mid, f0, f1, dt)


def _interp_evaluate(coefficients, t0, t1, t):
  """Evaluate polynomial interpolation at the given time point.

  Args:
    coefficients: list of Tensor coefficients as created by `interp_fit`.
    t0: scalar float64 Tensor giving the start of the interval.
    t1: scalar float64 Tensor giving the end of the interval.
    t: scalar float64 Tensor giving the desired interpolation point.

  Returns:
    Polynomial interpolation of the coefficients at time `t`.
  """
  with ops.name_scope('interp_evaluate'):
    t0 = ops.convert_to_tensor(t0)
    t1 = ops.convert_to_tensor(t1)
    t = ops.convert_to_tensor(t)

    dtype = coefficients[0].dtype

    assert_op = control_flow_ops.Assert(
        (t0 <= t) & (t <= t1),
        ['invalid interpolation, fails `t0 <= t <= t1`:', t0, t, t1])
    with ops.control_dependencies([assert_op]):
      x = math_ops.cast((t - t0) / (t1 - t0), dtype)

    xs = [constant_op.constant(1, dtype), x]
    for _ in range(2, len(coefficients)):
      xs.append(xs[-1] * x)

    return _dot_product(coefficients, reversed(xs))


def _optimal_step_size(last_step,
                       error_ratio,
                       safety=0.9,
                       ifactor=10.0,
                       dfactor=0.2,
                       order=5,
                       name=None):
  """Calculate the optimal size for the next Runge-Kutta step."""
  with ops.name_scope(
      name, 'optimal_step_size', [last_step, error_ratio]) as scope:
    error_ratio = math_ops.cast(error_ratio, last_step.dtype)
    exponent = math_ops.cast(1 / order, last_step.dtype)
    # this looks more complex than necessary, but importantly it keeps
    # error_ratio in the numerator so we can't divide by zero:
    factor = math_ops.maximum(
        1 / ifactor,
        math_ops.minimum(error_ratio ** exponent / safety, 1 / dfactor))
    return math_ops.div(last_step, factor, name=scope)


def _abs_square(x):
  if x.dtype.is_complex:
    return math_ops.square(math_ops.real(x)) + math_ops.square(math_ops.imag(x))
  else:
    return math_ops.square(x)


def _ta_append(tensor_array, value):
  """Append a value to the end of a tf.TensorArray."""
  return tensor_array.write(tensor_array.size(), value)


class _RungeKuttaState(collections.namedtuple(
    '_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')):
  """Saved state of the Runge Kutta solver.

  Attributes:
    y1: Tensor giving the function value at the end of the last time step.
    f1: Tensor giving derivative at the end of the last time step.
    t0: scalar float64 Tensor giving start of the last time step.
    t1: scalar float64 Tensor giving end of the last time step.
    dt: scalar float64 Tensor giving the size for the next time step.
    interp_coef: list of Tensors giving coefficients for polynomial
      interpolation between `t0` and `t1`.
  """


class _History(collections.namedtuple(
    '_History', 'integrate_points, error_ratio')):
  """Saved integration history for use in `info_dict`.

  Attributes:
    integrate_points: tf.TensorArray storing integrating time points.
    error_ratio: tf.TensorArray storing computed error ratios at each
      integration step.
  """


def _dopri5(func,
            y0,
            t,
            rtol,
            atol,
            full_output=False,
            first_step=None,
            safety=0.9,
            ifactor=10.0,
            dfactor=0.2,
            max_num_steps=1000,
            name=None):
  """Solve an ODE for `odeint` using method='dopri5'."""

  if first_step is None:
    # at some point, we might want to switch to picking the step size
    # automatically
    first_step = 1.0

  with ops.name_scope(
      name, 'dopri5',
      [y0, t, rtol, atol, safety, ifactor, dfactor, max_num_steps]) as scope:

    first_step = ops.convert_to_tensor(first_step, dtype=t.dtype,
                                       name='first_step')
    safety = ops.convert_to_tensor(safety, dtype=t.dtype, name='safety')
    ifactor = ops.convert_to_tensor(ifactor, dtype=t.dtype, name='ifactor')
    dfactor = ops.convert_to_tensor(dfactor, dtype=t.dtype, name='dfactor')
    max_num_steps = ops.convert_to_tensor(max_num_steps, dtype=dtypes.int32,
                                          name='max_num_steps')

    def adaptive_runge_kutta_step(rk_state, history, n_steps):
      """Take an adaptive Runge-Kutta step to integrate the ODE."""
      y0, f0, _, t0, dt, interp_coeff = rk_state
      with ops.name_scope('assertions'):
        check_underflow = control_flow_ops.Assert(
            t0 + dt > t0, ['underflow in dt', dt])
        check_max_num_steps = control_flow_ops.Assert(
            n_steps < max_num_steps, ['max_num_steps exceeded'])
        check_numerics = control_flow_ops.Assert(
            math_ops.reduce_all(math_ops.is_finite(abs(y0))),
            ['non-finite values in state `y`', y0])
      with ops.control_dependencies(
          [check_underflow, check_max_num_steps, check_numerics]):
        y1, f1, y1_error, k = _runge_kutta_step(func, y0, f0, t0, dt)

      with ops.name_scope('error_ratio'):
        # We use the same approach as the dopri5 fortran code.
        error_tol = atol + rtol * math_ops.maximum(abs(y0), abs(y1))
        tensor_error_ratio = _abs_square(y1_error) / _abs_square(error_tol)
        # Could also use reduce_maximum here.
        error_ratio = math_ops.sqrt(math_ops.reduce_mean(tensor_error_ratio))
        accept_step = error_ratio <= 1

      with ops.name_scope('update/rk_state'):
        # If we don't accept the step, the _RungeKuttaState will be useless
        # (covering a time-interval of size 0), but that's OK, because in such
        # cases we always immediately take another Runge-Kutta step.
        y_next = control_flow_ops.cond(accept_step, lambda: y1, lambda: y0)
        f_next = control_flow_ops.cond(accept_step, lambda: f1, lambda: f0)
        t_next = control_flow_ops.cond(accept_step, lambda: t0 + dt, lambda: t0)
        interp_coeff = control_flow_ops.cond(
            accept_step,
            lambda: _interp_fit_rk(y0, y1, k, dt),
            lambda: interp_coeff)
        dt_next = _optimal_step_size(dt, error_ratio, safety, ifactor, dfactor)
        rk_state = _RungeKuttaState(
            y_next, f_next, t0, t_next, dt_next, interp_coeff)

      with ops.name_scope('update/history'):
        history = _History(_ta_append(history.integrate_points, t0 + dt),
                           _ta_append(history.error_ratio, error_ratio))
      return rk_state, history, n_steps + 1

    def interpolate(solution, history, rk_state, i):
      """Interpolate through the next time point, integrating as necessary."""
      with ops.name_scope('interpolate'):
        rk_state, history, _ = control_flow_ops.while_loop(
            lambda rk_state, *_: t[i] > rk_state.t1,
            adaptive_runge_kutta_step,
            (rk_state, history, 0),
            name='integrate_loop')
        y = _interp_evaluate(
            rk_state.interp_coeff, rk_state.t0, rk_state.t1, t[i])
        solution = solution.write(i, y)
        return solution, history, rk_state, i + 1

    assert_increasing = control_flow_ops.Assert(
        math_ops.reduce_all(t[1:] > t[:-1]),
        ['`t` must be monotonic increasing'])
    with ops.control_dependencies([assert_increasing]):
      num_times = array_ops.size(t)

    solution = tensor_array_ops.TensorArray(
        y0.dtype, size=num_times).write(0, y0)
    history = _History(
        integrate_points=tensor_array_ops.TensorArray(
            t.dtype, size=0, dynamic_size=True),
        error_ratio=tensor_array_ops.TensorArray(
            rtol.dtype, size=0, dynamic_size=True))
    rk_state = _RungeKuttaState(
        y0, func(y0, t[0]), t[0], t[0], first_step, interp_coeff=[y0] * 5)

    solution, history, _, _ = control_flow_ops.while_loop(
        lambda _, __, ___, i: i < num_times,
        interpolate,
        (solution, history, rk_state, 1),
        name='interpolate_loop')

    y = solution.stack(name=scope)
    y.set_shape(t.get_shape().concatenate(y0.get_shape()))
    if not full_output:
      return y
    else:
      integrate_points = history.integrate_points.stack()
      info_dict = {'num_func_evals': 6 * array_ops.size(integrate_points) + 1,
                   'integrate_points': integrate_points,
                   'error_ratio': history.error_ratio.stack()}
      return (y, info_dict)


def odeint(func,
           y0,
           t,
           rtol=1e-6,
           atol=1e-12,
           method=None,
           options=None,
           full_output=False,
           name=None):
  """Integrate a system of ordinary differential equations.

  Solves the initial value problem for a non-stiff system of first order ode-s:

    ```
    dy/dt = func(y, t), y(t[0]) = y0
    ```

  where y is a Tensor of any shape.

  For example:

    ```
    # solve `dy/dt = -y`, corresponding to exponential decay
    tf.contrib.integrate.odeint(lambda y, _: -y, 1.0, [0, 1, 2])
    => [1, exp(-1), exp(-2)]
    ```

  Output dtypes and numerical precision are based on the dtypes of the inputs
  `y0` and `t`.

  Currently, implements 5th order Runge-Kutta with adaptive step size control
  and dense output, using the Dormand-Prince method. Similar to the 'dopri5'
  method of `scipy.integrate.ode` and MATLAB's `ode45`.

  Based on: Shampine, Lawrence F. (1986), "Some Practical Runge-Kutta Formulas",
  Mathematics of Computation, American Mathematical Society, 46 (173): 135-150,
  doi:10.2307/2008219

  Args:
    func: Function that maps a Tensor holding the state `y` and a scalar Tensor
      `t` into a Tensor of state derivatives with respect to time.
    y0: N-D Tensor giving starting value of `y` at time point `t[0]`. May
      have any floating point or complex dtype.
    t: 1-D Tensor holding a sequence of time points for which to solve for
      `y`. The initial time point should be the first element of this sequence,
      and each time must be larger than the previous time. May have any floating
      point dtype. If not provided as a Tensor, converted to a Tensor with
      float64 dtype.
    rtol: optional float64 Tensor specifying an upper bound on relative error,
      per element of `y`.
    atol: optional float64 Tensor specifying an upper bound on absolute error,
      per element of `y`.
    method: optional string indicating the integration method to use. Currently,
      the only valid option is `'dopri5'`.
    options: optional dict of configuring options for the indicated integration
      method. Can only be provided if a `method` is explicitly set. For
      `'dopri5'`, valid options include:
      * first_step: an initial guess for the size of the first integration
        (current default: 1.0, but may later be changed to use heuristics based
        on the gradient).
      * safety: safety factor for adaptive step control, generally a constant
        in the range 0.8-1 (default: 0.9).
      * ifactor: maximum factor by which the adaptive step may be increased
        (default: 10.0).
      * dfactor: maximum factor by which the adpative step may be decreased
        (default: 0.2).
      * max_num_steps: integer maximum number of integrate steps between time
        points in `t` (default: 1000).
    full_output: optional boolean. If True, `odeint` returns a tuple
      `(y, info_dict)` describing the integration process.
    name: Optional name for this operation.

  Returns:
    y: (N+1)-D tensor, where the first dimension corresponds to different
      time points. Contains the solved value of y for each desired time point in
      `t`, with the initial value `y0` being the first element along the first
      dimension.
    info_dict: only if `full_output == True`. A dict with the following values:
      * num_func_evals: integer Tensor counting the number of function
        evaluations.
      * integrate_points: 1D float64 Tensor with the upper bound of each
        integration time step.
      * error_ratio: 1D float Tensor with the estimated ratio of the integration
        error to the error tolerance at each integration step. An ratio greater
        than 1 corresponds to rejected steps.

  Raises:
    ValueError: if an invalid `method` is provided.
    TypeError: if `options` is supplied without `method`, or if `t` or `y0` has
      an invalid dtype.
  """
  if method is not None and method != 'dopri5':
    raise ValueError('invalid method: %r' % method)

  if options is None:
    options = {}
  elif method is None:
    raise ValueError('cannot supply `options` without specifying `method`')

  with ops.name_scope(name, 'odeint', [y0, t, rtol, atol]) as scope:
    # TODO(shoyer): use nest.flatten (like tf.while_loop) to allow `y0` to be an
    # arbitrarily nested tuple. This will help performance and usability by
    # avoiding the need to pack/unpack in user functions.
    y0 = ops.convert_to_tensor(y0, name='y0')
    if not (y0.dtype.is_floating or y0.dtype.is_complex):
      raise TypeError('`y0` must have a floating point or complex floating '
                      'point dtype')

    t = ops.convert_to_tensor(t, preferred_dtype=dtypes.float64, name='t')
    if not t.dtype.is_floating:
      raise TypeError('`t` must have a floating point dtype')

    error_dtype = abs(y0).dtype
    rtol = ops.convert_to_tensor(rtol, dtype=error_dtype, name='rtol')
    atol = ops.convert_to_tensor(atol, dtype=error_dtype, name='atol')

    return _dopri5(func, y0, t,
                   rtol=rtol,
                   atol=atol,
                   full_output=full_output,
                   name=scope,
                   **options)
