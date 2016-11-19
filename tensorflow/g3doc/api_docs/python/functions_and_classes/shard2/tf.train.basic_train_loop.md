### `tf.train.basic_train_loop(supervisor, train_step_fn, args=None, kwargs=None, master='')` {#basic_train_loop}

Basic loop to train a model.

Calls `train_step_fn` in a loop to train a model.  The function is called as:

```python
train_step_fn(session, *args, **kwargs)
```

It is passed a `tf.Session` in addition to `args` and `kwargs`.  The function
typically runs one training step in the session.

##### Args:


*  <b>`supervisor`</b>: `tf.Supervisor` to run the training services.
*  <b>`train_step_fn`</b>: Callable to execute one training step.  Called
    repeatedly as `train_step_fn(session, *args **kwargs)`.
*  <b>`args`</b>: Optional positional arguments passed to `train_step_fn`.
*  <b>`kwargs`</b>: Optional keyword arguments passed to `train_step_fn`.
*  <b>`master`</b>: Master to use to create the training session.  Defaults to
    `""` which causes the session to be created in the local process.

