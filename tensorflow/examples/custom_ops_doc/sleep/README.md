# Create an asynchronous sleep op

This guide provides an end-to-end example for an asynchronous custom op. The
example implements an asynchronous sleep op, and contrasts the implementation
with a synchronous sleep op.

Asynchronous ops allow other ops to execute while the asynchronous op waits.
Unlike synchronous ops, an asynchronous op does not block other ops as it waits.

The content on this page assumes familiarity with the high-level process for
adding custom ops to TensorFlow. For additional context,
read the
[OSS guide on creating custom ops](https://www.tensorflow.org/guide/create_op).

## Background information on asynchronous ops

Asynchronous ops are recommended for cases where the number of ops that may be
waiting at a given time is significantly larger than the desired number of
threads. In general, you should use asynchronous ops for most cases where an op
waits, though for specific cases (e.g. short waits) the overhead and complexity
of an asynchronous op may outweigh the benefits.

You can also use event and queuing based techniques, rather than using threads.
For cases where the op interfaces to something that already uses a callback, an
asynchronous op implementation is more straightforward than the alternatives.

Asynchronous ops are derived from
[`AsyncOpKernel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#L225)
and use the `ComputeAsync()` method to override the default `Compute()` method
used by synchronous ops
([`OpKernel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#L104)).

An op can delegate the task of producing a result to another function, method,
or closure using `ComputeAsync()`. The op can then return before all the work
has completed. For example, the op can schedule a closure to run in another
thread and then return immediately.

### The `done` callback function

The `ComputeAsync()` method contains a `done` parameter, which is passed to
other functions as a callback function. Once the other function sets the output
or completes the work, it calls `done` to notify that the op has finished.

The `done` function must be called exactly once in every execution path. Any
paths in `ComputeAsync()` that return early, whether due to error handling or
cases where results are produced quickly, must call `done`. Similarly, if the
function that receives the `done` callback has any execution paths that return
early, the function must call `done` in these paths in addition to calling
`done` before it returns normally.

Once `done` is triggered, any ops that depend on the output(s) of this op can
execute.

<!-- test_snippets_in_readme skip -->
```c++
class MyAsyncOp : public AsyncOpKernel {
 // …
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // …
    OP_REQUIRES_OK_ASYNC(ctx, ErrorChecking(), done);
    // …
    // ScheduleOrRequest is pseudocode for scheduling or requesting work elsewhere
    ScheduleOrRequest(delegate, done /* , … */ );
    // return quickly without waiting for the result
  }

}
void delegate(DoneCallback done /* , … */ ) {
  // set output of op and/or do work of op
  // …
  done();
}
```

#### Alternatives for parallelizing computation

While this example implements an op that waits, there are other cases where you
may seek an alternative to executing computationally-intensive ops in parallel.
For example:

*   Ops in `tf.function` graphs can be executed in parallel by multiple threads
    from the inter-op thread pool. You can use synchronous ops when there are at
    least as many threads as ops that can run at the same time. Even if there
    are fewer threads than ops, this may still be a sufficient implementation.
*   `tf.data` can be used for cases that create an input pipe.
    `tf.data.dataset.map()` can be used with a synchronous custom op, where the
    `num_parallel_calls` parameter is used to run that op in parallel with a
    thread pool provided by `tf.data`. For more information, see
    [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance)
    and
    [Preprocessing data](https://www.tensorflow.org/guide/data#preprocessing_data).
*   [tf.map_fn](https://www.tensorflow.org/api_docs/python/tf/map_fn) and
    [tf.while_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop)
    can be used with a synchronous custom op and the `parallel_iterations`
    parameter to run that op in parallel using multiple threads.

## Creating an asynchronous sleep op

This example demonstrates how you can create an asynchronous sleep op,
`sleep_op`, that waits for a specified amount of time while letting other ops
run. While this example does not wait for any particular purpose, it illustrates
the pattern for ops that delay or poll while waiting for something, such as
another op's result.

For contrast, the example also includes a synchronous version of the same sleep
op.

The following example waits for one second with the synchronous and asynchronous
versions:

<!-- test_snippets_in_readme skip -->
```python
sleep_op.SyncSleep(1.0)
# tf.Tensor(1.0, shape=(), dtype=float32)
sleep_op.AsyncSleep(1.0)
# tf.Tensor(0.999892, shape=(), dtype=float32)
```

The synchronous version (`sleep_op.SyncSleep`) simply blocks time by calling
`absl::SleepFor` from
[clock.h](https://github.com/abseil/abseil-cpp/blob/8c6e53ef3adb1227fffa442c50349dab134a54bc/absl/time/clock.h)
(similar to `sleep`) for the amount of time specified in the input (1 second),
and returns that delay value in the output.

The asynchronous version (`sleep_op.AsyncSleep`) uses the input (1 second) and
the current time to compute the wake-up time, which is the point in time when
the function stops waiting. The op schedules a function that receives this
wake-up time to run in a threadpool, and returns immediately.

The function either begins running immediately after it is scheduled or after
blocking for some time. If the current time is before the wake-up time, the
function computes the difference and blocks time using `absl::SleepFor`.
Otherwise, if the current time is equal to or after the wake-up time, it does
not call `absl::SleepFor`. The function sets the output to the time specified
for sleep, or 0 if it does not call `absl::SleepFor`, and calls the `done`
callback.

In the example above, the asynchronous function called sleep for 0.999892
seconds after being blocked for 0.000108 seconds waiting for the function to
run. This waits for a total time of 1 second, as specified in the input.

This example contains C++ and Python code snippets to illustrate the code flow.
These snippets may be missing namespace declarations, imports, and test cases.

### Step 1 - Define the op interface

Define the op interface and register it using the `REGISTER_OP` macro.

```
REGISTER_OP("Examples>AsyncSleep")
    .Input("delay: float")
    .Output("output: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      return (ScalarOutput(c));
    })
    .Doc(R"doc(
Pause for `delay` seconds (which need not be an integer).

This is an asynchronous (non-blocking) version of sleep. It is intended to
be an example of how to implements ops that do I/O or that block on other ops.

delay: tf.Tensor which is a scalar of type float.

Returns the time spent in blocking sleep (which may be less that `delay` or
zero if other ops run while this is waiting asynchronously).
)doc");
```
```
REGISTER_OP("Examples>SyncSleep")
    .Input("delay: float")
    .Output("output: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      return (ScalarOutput(c));
    })
    .Doc(R"doc(
Pause for `delay` seconds (which need not be an integer).

This is a synchronous (blocking) version of sleep. It's purpose is
to be contrasted with Examples>AsyncSleep.

delay: tf.Tensor which is a scalar of type float.

Returns `delay`.
)doc");
```

The op registers two examples, an asynchronous op (`Examples>AsyncSleep`) and a
synchronous op (`Examples>SyncSleep`). Both examples accept `delay` as an input,
which is a scalar of type float. The `AsyncSleep` example returns the time spent
in blocking sleep (which may be less than `delay` if other ops run while the op
is waiting asynchronously), while the `SyncSleep` example simply returns the
value of `delay`.

### Step 2 - Register the op implementation (kernel)

The C++ kernel in
`sleep_kernel.cc`
implements both a synchronous (`SyncSleepOp`) and asynchronous (`AsyncSleepOp`)
sleep op.

<!-- test_snippets_in_readme skip -->
```c++
REGISTER_KERNEL_BUILDER(
    Name("Examples>AsyncSleep").Device(::tensorflow::DEVICE_CPU), AsyncSleepOp)
REGISTER_KERNEL_BUILDER(
    Name("Examples>SyncSleep").Device(::tensorflow::DEVICE_CPU), SyncSleepOp)
```

### Step 3 - Implement the op kernel(s)

In the `sleep_kernel.cc` op kernel, create two classes: one derived from
`OpKernel` and another derived from `AsyncOpKernel`.

The `OpKernel` class will be familiar if you have followed the other custom op
examples in this series. It implements a `Compute()` method, which is used for
the synchronous sleep op. The new `AsyncOpKernel` class uses `ComputeAsync()`,
which is used for the asynchronous sleep op.

```
class AsyncSleepOp : public AsyncOpKernel {
 public:
  explicit AsyncSleepOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {}
  AsyncSleepOp(const AsyncSleepOp& other) = delete;
  AsyncSleepOp& operator=(const AsyncSleepOp& other) = delete;
  ~AsyncSleepOp() override = default;

  // Implementations of ComputeAsync() must ensure that `done` is (eventually)
  // called exactly once to signal the completion of the computation. The
  // implementation of ComputeAsync() must not block on the execution of another
  // OpKernel. `done` may be called by the current thread, or by another thread.
  // `context` is guaranteed to stay alive until the `done` callback starts.
  // For example, use OP_REQUIRES_ASYNC which takes the `done` paramater
  // as an input and calls `done` for the case of exiting early with an error
  // (instead of OP_REQUIRES).
  //
  // Since it is possible that the unblocking kernel may never run (due to an
  // error or cancellation), in most cases the AsyncOpKernel should implement
  // cancellation support via `context->cancellation_manager()`.
  // TODO (schwartzedward): should this use cancellation support?
  //
  // WARNING: As soon as the `done` callback starts, `context` and `this` may be
  // deleted. No code depending on these objects should execute after the call
  // to `done`.
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const auto& delay_tensor = ctx->input(0);
    OP_REQUIRES_ASYNC(
        ctx, ::tensorflow::TensorShapeUtils::IsScalar(delay_tensor.shape()),
        InvalidArgument("Input `delay` must be a scalar."),
        done);  // Important: call `done` in every execution path
    const float delay = delay_tensor.flat<float>()(0);
    OP_REQUIRES_ASYNC(ctx, delay >= 0.0,
                      InvalidArgument("Input `delay` must be non-negative."),
                      done);  // Important: call `done` in every execution path
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    OP_REQUIRES_ASYNC(ctx, thread_pool != nullptr,
                      Internal("No thread_pool found."),
                      done);  // Important: call `done` in every execution path

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, delay_tensor.shape(), &output_tensor),
        done);  // Important: call `done` in every execution path

    absl::Time now = absl::Now();
    absl::Time when = now + absl::Seconds(delay);
    VLOG(1) << "BEFORE ASYNC SLEEP " << ctx->op_kernel().name() << " now "
            << now << " when " << when;
    thread_pool->Schedule([this, output_tensor, when, done] {
      this->sleeper(output_tensor, when, done);
    });
    // Note that `done` is normaly called by sleeper(), it is not normally
    // called by this function.
  }

 private:
  void sleeper(Tensor* output_tensor, absl::Time when, DoneCallback done) {
    absl::Time now = absl::Now();
    int64_t delay_us = 0;
    if (now < when) {
      delay_us = absl::ToInt64Microseconds(when - now);
      VLOG(1) << "MIDDLE ASYNC SLEEP " << delay_us;
      absl::SleepFor(when - now);
      VLOG(1) << "AFTER ASYNC SLEEP " << delay_us;
    } else {
      VLOG(1) << "MIDDLE/AFTER ASYNC SKIP SLEEP";
    }
    auto output = output_tensor->template flat<float>();
    output(0) = static_cast<float>(delay_us) / 1000000.0;
    done();  // Important: call `done` in every execution path
  }
};
```

The implementation overrides `ComputeAsync`, which has a `done` callback
argument. For error checking, it uses `OP_REQUIRES_ASYNC` and
`OP_REQUIRES_OK_ASYNC`, which calls `done` if returning with an error.

Note: For asynchronous ops, always use `OP_REQUIRES_ASYNC` and
`OP_REQUIRES_OK_ASYNC`. Asynchronous ops cannot use `OP_REQUIRES` or
`OP_REQUIRES_OK`.

The wake-up time for the op is computed using `absl::Now` from
[clock.h](https://github.com/abseil/abseil-cpp/blob/8c6e53ef3adb1227fffa442c50349dab134a54bc/absl/time/clock.h).
The sleeper helper method receives the wake-up time and `done` callback, and
runs in the `tensorflow_cpu_worker_threads` thread pool.

For normal operations, `done` is called by the sleeper method, not by
`ComputeAsync`.

```
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const auto& delay_tensor = ctx->input(0);
    OP_REQUIRES_ASYNC(
        ctx, ::tensorflow::TensorShapeUtils::IsScalar(delay_tensor.shape()),
        InvalidArgument("Input `delay` must be a scalar."),
        done);  // Important: call `done` in every execution path
    const float delay = delay_tensor.flat<float>()(0);
    OP_REQUIRES_ASYNC(ctx, delay >= 0.0,
                      InvalidArgument("Input `delay` must be non-negative."),
                      done);  // Important: call `done` in every execution path
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    OP_REQUIRES_ASYNC(ctx, thread_pool != nullptr,
                      Internal("No thread_pool found."),
                      done);  // Important: call `done` in every execution path

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, delay_tensor.shape(), &output_tensor),
        done);  // Important: call `done` in every execution path

    absl::Time now = absl::Now();
    absl::Time when = now + absl::Seconds(delay);
    VLOG(1) << "BEFORE ASYNC SLEEP " << ctx->op_kernel().name() << " now "
            << now << " when " << when;
    thread_pool->Schedule([this, output_tensor, when, done] {
      this->sleeper(output_tensor, when, done);
    });
    // Note that `done` is normaly called by sleeper(), it is not normally
    // called by this function.
  }
```

If the current time is before the specified wake-up time, the `sleeper` helper
method sleeps for the difference between the current time and wake-up time. It
sets the output and calls the `done` callback, which notifies the op that the
output is set.

Calling `done` also invalidates `ctx`, so its contents cannot be used after
calling `done`.

```
  void sleeper(Tensor* output_tensor, absl::Time when, DoneCallback done) {
    absl::Time now = absl::Now();
    int64_t delay_us = 0;
    if (now < when) {
      delay_us = absl::ToInt64Microseconds(when - now);
      VLOG(1) << "MIDDLE ASYNC SLEEP " << delay_us;
      absl::SleepFor(when - now);
      VLOG(1) << "AFTER ASYNC SLEEP " << delay_us;
    } else {
      VLOG(1) << "MIDDLE/AFTER ASYNC SKIP SLEEP";
    }
    auto output = output_tensor->template flat<float>();
    output(0) = static_cast<float>(delay_us) / 1000000.0;
    done();  // Important: call `done` in every execution path
  }
```

#### Compile the op

Compile the C++ op to create a kernel library and Python wrapper that enables
you to use the op with TensorFlow.

The `BUILD` file declares the dependencies and the output build targets. Refer to
[building for OSS](https://www.tensorflow.org/guide/create_op#build_the_op_library).

You will be reusing the `BUILD` file later in this example.

```
tf_custom_op_library(
    name = "sleep_kernel.so",
    srcs = [
        "sleep_kernel.cc",
        "sleep_op.cc",
    ],
    deps = [
        "//third_party/absl/time",
    ],
)

py_strict_library(
    name = "sleep_op",
    srcs = ["sleep_op.py"],
    data = ["sleep_kernel.so"],
    srcs_version = "PY3",
)

py_strict_binary(
    name = "sleep_bin",
    srcs = ["sleep_bin.py"],
    srcs_version = "PY3",
    deps = [
        ":sleep_op",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "@absl_py//absl:app",
    ],
)
```

### Step 4 - Create the Python wrapper

To create the Python wrapper, import and implement a function that serves as the
op's public API and provides a docstring.

```
def AsyncSleep(delay, name=None):
  """Pause for `delay` seconds (which need not be an integer).

  This is an asynchronous (non-blocking) version of a sleep op. It includes
  any time spent being blocked by another thread in `delay`. If it is blocked
  for a fraction of the time specified by `delay`, it only calls `sleep`
  (actually `usleep`) only for the remainder. If it is blocked for the full
  time specified by `delay` or more, it returns without explictly calling
  `sleep`.

  Args:
    delay: tf.Tensor which is a scalar of type float.
    name: An optional name for the op.

  Returns:
    The `delay` value.
  """
  return gen_sleep_op.examples_async_sleep(delay=delay, name=name)
```

```
def SyncSleep(delay, name=None):
  """Pause for `delay` seconds (which need not be an integer).

  This is a synchronous (blocking) version of a sleep op. It's purpose is
  to be contrasted with Examples>AsyncSleep.

  Args:
    delay: tf.Tensor which is a scalar of type float.
    name: An optional name for the op.

  Returns:
    The `delay` value.
  """
  return gen_sleep_op.examples_sync_sleep(delay=delay, name=name)
```

### Step 5 - Test the op

Create op tests using classes derived from
[`tf.test.TestCase`](https://www.tensorflow.org/api_docs/python/tf/test/TestCase).

When writing tests to ensure that the op works correctly in both graph and eager
executions, it is important to note that errors in the op code may be detected
in two distinct phases of code execution depending on how it is executed (eager
or graph). Errors may be detected early by the shape function or a
bit later from the logic in the `Compute` and `ComputeAsync` methods. This may
lead to differing error types and/or messages.

```
class SleepTest(tf.test.TestCase):

  def _check_sleep(self, op):
    """Check that one sleep op works in isolation.

    See sleep_bin.py for an example of how the synchronous and asynchronous
    sleep ops differ in behavior.

    Args:
      op: The sleep op, either sleep_op.SyncSleep or sleep_op.AsyncSleep.
    """
    delay = 0.3  # delay in seconds
    start_t = time.time()
    func = tf.function(lambda: op(delay))
    results = self.evaluate(func())
    end_t = time.time()
    delta_t = end_t - start_t
    self.assertEqual(results.shape, tuple())
    self.assertGreater(delta_t, 0.9 * delay)

  def test_sync_sleep(self):
    self._check_sleep(sleep_op.SyncSleep)

  def test_async_sleep(self):
    self._check_sleep(sleep_op.AsyncSleep)

  def test_async_sleep_error(self):
    # It is import that ComputeAsync() calls its done() callback if it returns
    # early due to an error.
    func = tf.function(lambda: sleep_op.AsyncSleep(-1.0))
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                'Input `delay` must be non-negative.'):
      self.evaluate(func())
```

The tests in `sleep_test.py` only tests a single sleep op in isolation, so they
do not demonstrate the difference in behavior between the asynchronous and
synchronous versions. Ideally, tests should cover every codepath to confirm that
`done` is always called.

You can test the op and create a build that shows the difference between
synchronous and asynchronous behavior. After running the `sleep_bin.py` binary,
the output will look something like this:

<!-- test_snippets_in_readme skip -->
```
Using synchronous sleep op with each of 50 ops
sleeping for about 1.00 seconds, so total time is about 1.00 * ceil(50 /
NUMBER_OF_THREADS). 16 is a typical number of threads, which would be 4.00
seconds. The actual time will be a little greater.

Total time = 4.555 seconds using <function SyncSleep at 0x7f19a401b4d0> Returned
values from the ops:

[1. 1.0001 1.0002 1.0003 1.0004 1.0005 1.0006 1.0007 1.0008 1.0009 1.001 1.0011
1.0012 1.0013 1.0014 1.0015 1.0016 1.0017 1.0018 1.0019 1.002 1.0021 1.0022
1.0023 1.0024 1.0025 1.0026 1.0027 1.0028 1.0029 1.003 1.0031 1.0032 1.0033
1.0034 1.0035 1.0036 1.0037 1.0038 1.0039 1.004 1.0041 1.0042 1.0043 1.0044
1.0045 1.0046 1.0047 1.0048 1.0049]

Using asynchronous sleep op with each of 50 ops sleeping only as much as
necessary so they finish after at least 1.00 seconds. Time that an op spends
blocked waiting to finish counts as all or part of its delay. The returned
values show how long each ops sleeps or 0 if the op does not need to sleep. The
expected total time will be a little greater than the requested delay of 1.00
seconds.

Total time = 1.450 seconds using <function AsyncSleep at 0x7f19a401b170>
Returned values from the ops: [0. 0. 0. 0. 0. 1.0001 1.0005 0. 0. 1.0008 0. 0.
1.0011 1.0013 0. 0. 0.0004 0.0012 1.0018 0.0001 1.002 0.001 0. 0.0006 0. 0.
0.0005 0. 0.001 0. 0. 1.003 1.0031 0.0003 0.0005 0.0033 1.0032 0.0004 1.0037
1.0038 0.0014 0.0023 0.0008 0.0024 0.0014 1.0044 1.0045 1.0046 1.0047 0.004 ]
```

The returned values from the synchronous ops are the same as the input delay
values, which are all different but close to 1 second. The returned values from
the asynchronous ops are the time blocked using `absl::SleepFor`, or 0 if the op
did not sleep.

In this example, 16 of the 50 ops slept for approximately the entire requested
time, 17 of the 50 ops slept for a fraction of the time, and 17 did not sleep at
all.

### Use the op

Use sleep_bin.py to explore the differences between synchronous and asynchronous
behavior.

The file creates a
[`tf.stack`](https://www.tensorflow.org/api_docs/python/tf/stack) of 50 sleep
custom ops that can potentially run in parallel. Each op receives a slightly
different input argument so that the ops are not combined as common
subexpressions.

Note: Multiple sleep ops with different inputs can be used in one `tf.function`
without being combined by optimization.

```
def stack50(op, delay):
  """Create a tf.stack of 50 sleep ops.

  Args:
    op: The sleep op, either sleep_op.SyncSleep or sleep_op.AsyncSleep.
    delay: Each op should finish at least float `delay` seconds after it starts.
  """
  n = 50
  delays = delay + tf.range(0, n, dtype=float) / 10000.0
  start_t = time.time()
  func = tf.function(lambda: tf.stack([op(delays[i]) for i in range(n)]))
  r_numpy = func().numpy()
  end_t = time.time()
  print('')
  print('Total time = %5.3f seconds using %s' % (end_t - start_t, str(op)))
  print('Returned values from the ops:')
  np.set_printoptions(precision=4, suppress=True)
  print(r_numpy)
  sys.stdout.flush()
```

When `stack50` is called with the synchronous op, each of the 50 ops sleeps
independently. The number of threads used to execute ops is configurable. With
the default of 16 threads, the total time is a little greater than 4 seconds
(ceil(50/16) = 4, with a delay of 1 second).

<!-- test_snippets_in_readme skip -->
```python
  delay_seconds = 1.0
  stack50(sleep_op.SyncSleep, delay_seconds)
```

When `stack50` is called with the asynchronous op, each op considers any time it
spends blocked waiting to be scheduled in a thread pool as part of its delay,
and only sleeps for the remaining time. For a delay of 1 second, the total
time is a little greater than 1 second (regardless of how many ops are sleeping
or how many threads are in the thread pool).

<!-- test_snippets_in_readme skip -->
```python
  delay_seconds = 1.0
  stack50(sleep_op.AsyncSleep, delay_seconds)
```

A more appropriate solution for actual delay or polling use cases is to
use queuing or scheduling, where one or more asynchronous ops manage multiple
delays or polls.

### Summary

In this example, you learned how to implement a synchronous and asynchronous
custom op for GPU. Using a helper method, you implemented an asynchronous sleep
op.

The table below summarizes the build rules and targets for building and testing
the `sleep` op.

| Op components  | Build rule             | Build target   | Source            |
| -------------- | ---------------------- | -------------- | ----------------- |
| Kernels (C++)  | `tf_kernel_library`    | `sleep_kernel` | `sleep_kernel.cc` |
| Wrapper        | `tf_gen_op_wrapper.py` | `gen_sleep_op` | N/A               |
: (automatically :                        :                :                   :
: generated)     :                        :                :                   :
| Wrapper (with  | `py_strict_library`    | `sleep_op`     | `sleep_op.py`     |
: public API and :                        :                :                   :
: docstring)     :                        :                :                   :
| Tests          | `tf_py_test`           | `sleep_test`   | `sleep_test.py`   |
| Example        | `py_strict_binary`     | `sleep_bin`    | `sleep_bin.py`    |

