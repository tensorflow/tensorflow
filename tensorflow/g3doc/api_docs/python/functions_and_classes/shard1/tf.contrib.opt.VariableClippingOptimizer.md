Wrapper optimizer that clips the norm of specified variables after update.

This optimizer delegates all aspects of gradient calculation and application
to an underlying optimizer.  After applying gradients, this optimizer then
clips the variable to have a maximum L2 norm along specified dimensions.
NB: this is quite different from clipping the norm of the gradients.

Multiple instances of `VariableClippingOptimizer` may be chained to specify
different max norms for different subsets of variables.

This is more efficient at serving-time than using normalization during
embedding lookup, at the expense of more expensive training and fewer
guarantees about the norms.

- - -

#### `tf.contrib.opt.VariableClippingOptimizer.__init__(opt, vars_to_clip_dims, max_norm, use_locking=False, colocate_clip_ops_with_vars=False, name='VariableClipping')` {#VariableClippingOptimizer.__init__}

Construct a new clip-norm optimizer.

##### Args:


*  <b>`opt`</b>: The actual optimizer that will be used to compute and apply the
    gradients. Must be one of the Optimizer classes.
*  <b>`vars_to_clip_dims`</b>: A dict with keys as Variables and values as lists
    of dimensions along which to compute the L2-norm.  See
    `tf.clip_by_norm` for more details.
*  <b>`max_norm`</b>: The L2-norm to clip to, for all variables specified.
*  <b>`use_locking`</b>: If `True` use locks for clip update operations.
*  <b>`colocate_clip_ops_with_vars`</b>: If `True`, try colocating the clip norm
    ops with the corresponding variable.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "VariableClipping".



#### Other Methods
- - -

#### `tf.contrib.opt.VariableClippingOptimizer.apply_gradients(grads_and_vars, global_step=None, name=None)` {#VariableClippingOptimizer.apply_gradients}




- - -

#### `tf.contrib.opt.VariableClippingOptimizer.compute_gradients(*args, **kwargs)` {#VariableClippingOptimizer.compute_gradients}




- - -

#### `tf.contrib.opt.VariableClippingOptimizer.get_slot(*args, **kwargs)` {#VariableClippingOptimizer.get_slot}




- - -

#### `tf.contrib.opt.VariableClippingOptimizer.get_slot_names(*args, **kwargs)` {#VariableClippingOptimizer.get_slot_names}




