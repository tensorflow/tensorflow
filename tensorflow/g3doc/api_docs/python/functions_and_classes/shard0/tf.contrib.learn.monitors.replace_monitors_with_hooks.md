### `tf.contrib.learn.monitors.replace_monitors_with_hooks(monitors_or_hooks, estimator)` {#replace_monitors_with_hooks}

Wraps monitors with a hook.

`Monitor` is deprecated in favor of `SessionRunHook`. If you're using a
monitor, you can wrap it with a hook using function. It is recommended to
implement hook version of your monitor.

##### Args:


*  <b>`monitors_or_hooks`</b>: A `list` may contain both monitors and hooks.
*  <b>`estimator`</b>: An `Estimator` that monitor will be used with.

##### Returns:

  Returns a list of hooks. If there is any monitor in the given list, it is
  replaced by a hook.

