## Auto Detect and Advise

tfprof analyzes profiles and generates advises for common issues.

### Run Advise.
```python
# First create a profiler. See profiler tutorials for more details.
profiler = model_analyzer.Profiler(sess.graph)
run_meta = config_pb2.RunMetadata()
_ = sess.run(r1,
             options=config_pb2.RunOptions(
                 trace_level=config_pb2.RunOptions.FULL_TRACE),
             run_metadata=run_meta)
profiler.add_step(1, run_meta)

# Start advise.
profiler.advise()
```

### Checker

There is no magic behind advise mode. tfprof builds the profiles first, then
it runs through a list of `Checkers`, each one responsible for checking one
area with the profile and report issues. A `Checker` is like a plugin.

For example:

####JobChecker (Not Available OSS)
* Checking RecvTensor RPC latency and bandwidth.
* Checking CPU/Memory utilization of the job.

####AcceleratorUtilization Checker
* Checks what percentage of time the accelerator spends on computation.

####Operation Checker
* Check whether the operation runs with optimal options.
* Checks if there is a better implementation to replace the current operation.

####Contribute Your Checker

Follow examples of accelerator_utilization_checker.h



