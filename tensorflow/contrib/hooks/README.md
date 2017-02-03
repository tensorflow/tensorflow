# TensorFlow Experimental SessionRunHooks

These hooks complement those in tensorflow/python/training. They are instances
of `SessionRunHook` and are to be used with helpers like `MonitoredSession`
and `learn.Estimator` that wrap `tensorflow.Session`.

The hooks are called between invocations of `Session.run()` to perform custom
behaviour.

For example the `ProfilerHook` periodically collects `RunMetadata` after
`Session.run()` and saves profiling information that can be viewed in a
neat timeline through a Chromium-based web browser (via
[about:tracing](chrome://tracing)) or the standalone [Catapult](https://github.com/catapult-project/catapult/blob/master/tracing/README.md) tool.

```python
from tensorflow.contrib.hooks import ProfilerHook

hooks = [ProfilerHook(save_secs=30, output_dir="profiling")]
with SingularMonitoredSession(hooks=hooks) as sess:
  while not sess.should_stop():
    sess.run(some_op)
```

Or similarly with contrib.learn:

```python
hooks = [ProfilerHook(save_steps=10, output_dir="profiling")]
estimator = learn.Estimator(...)
estimator.fit(input_fn, monitors=hooks)
```
