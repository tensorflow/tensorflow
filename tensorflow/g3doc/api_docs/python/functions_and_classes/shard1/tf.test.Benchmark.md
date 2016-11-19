Abstract class that provides helpers for TensorFlow benchmarks.
- - -

#### `tf.test.Benchmark.is_abstract(cls)` {#Benchmark.is_abstract}




- - -

#### `tf.test.Benchmark.report_benchmark(iters=None, cpu_time=None, wall_time=None, throughput=None, extras=None, name=None)` {#Benchmark.report_benchmark}

Report a benchmark.

##### Args:


*  <b>`iters`</b>: (optional) How many iterations were run
*  <b>`cpu_time`</b>: (optional) Total cpu time in seconds
*  <b>`wall_time`</b>: (optional) Total wall time in seconds
*  <b>`throughput`</b>: (optional) Throughput (in MB/s)
*  <b>`extras`</b>: (optional) Dict mapping string keys to additional benchmark info.
    Values may be either floats or values that are convertible to strings.
*  <b>`name`</b>: (optional) Override the BenchmarkEntry name with `name`.
    Otherwise it is inferred from the top-level method name.


- - -

#### `tf.test.Benchmark.run_op_benchmark(sess, op_or_tensor, feed_dict=None, burn_iters=2, min_iters=10, store_trace=False, name=None, extras=None, mbs=0)` {#Benchmark.run_op_benchmark}

Run an op or tensor in the given session.  Report the results.

##### Args:


*  <b>`sess`</b>: `Session` object to use for timing.
*  <b>`op_or_tensor`</b>: `Operation` or `Tensor` to benchmark.
*  <b>`feed_dict`</b>: A `dict` of values to feed for each op iteration (see the
    `feed_dict` parameter of `Session.run`).
*  <b>`burn_iters`</b>: Number of burn-in iterations to run.
*  <b>`min_iters`</b>: Minimum number of iterations to use for timing.
*  <b>`store_trace`</b>: Boolean, whether to run an extra untimed iteration and
    store the trace of iteration in the benchmark report.
    The trace will be stored as a string in Google Chrome trace format
    in the extras field "full_trace_chrome_format".
*  <b>`name`</b>: (optional) Override the BenchmarkEntry name with `name`.
    Otherwise it is inferred from the top-level method name.
*  <b>`extras`</b>: (optional) Dict mapping string keys to additional benchmark info.
    Values may be either floats or values that are convertible to strings.
*  <b>`mbs`</b>: (optional) The number of megabytes moved by this op, used to
    calculate the ops throughput.

##### Returns:

  A `dict` containing the key-value pairs that were passed to
  `report_benchmark`.


