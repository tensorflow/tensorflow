Base class for interfaces with external optimization algorithms.

Subclass this and implement `_minimize` in order to wrap a new optimization
algorithm.

`ExternalOptimizerInterface` should not be instantiated directly; instead use
e.g. `ScipyOptimizerInterface`.

- - -

#### `tf.contrib.opt.ExternalOptimizerInterface.__init__(loss, var_list=None, equalities=None, inequalities=None, **optimizer_kwargs)` {#ExternalOptimizerInterface.__init__}

Initialize a new interface instance.

##### Args:


*  <b>`loss`</b>: A scalar `Tensor` to be minimized.
*  <b>`var_list`</b>: Optional list of `Variable` objects to update to minimize
    `loss`.  Defaults to the list of variables collected in the graph
    under the key `GraphKeys.TRAINABLE_VARIABLES`.
*  <b>`equalities`</b>: Optional list of equality constraint scalar `Tensor`s to be
    held equal to zero.
*  <b>`inequalities`</b>: Optional list of inequality constraint scalar `Tensor`s
    to be kept nonnegative.
*  <b>`**optimizer_kwargs`</b>: Other subclass-specific keyword arguments.



- - -

#### `tf.contrib.opt.ExternalOptimizerInterface.minimize(session=None, feed_dict=None, fetches=None, step_callback=None, loss_callback=None)` {#ExternalOptimizerInterface.minimize}

Minimize a scalar `Tensor`.

Variables subject to optimization are updated in-place at the end of
optimization.

Note that this method does *not* just return a minimization `Op`, unlike
`Optimizer.minimize()`; instead it actually performs minimization by
executing commands to control a `Session`.

##### Args:


*  <b>`session`</b>: A `Session` instance.
*  <b>`feed_dict`</b>: A feed dict to be passed to calls to `session.run`.
*  <b>`fetches`</b>: A list of `Tensor`s to fetch and supply to `loss_callback`
    as positional arguments.
*  <b>`step_callback`</b>: A function to be called at each optimization step;
    arguments are the current values of all optimization variables
    flattened into a single vector.
*  <b>`loss_callback`</b>: A function to be called every time the loss and gradients
    are computed, with evaluated fetches supplied as positional arguments.


