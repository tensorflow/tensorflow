# tfprof: TensorFlow Profiler and Beyond

<h1>Please use `tf.profiler.xxx` instead of `tf.contrib.tfprof.xxx`</h1>
<h1>Full Document in tensorflow/core/profiler/README.md<h1>

###Features

* Profile model architectures
  * parameters, tensor shapes, float operations, device placement, etc.
* Profile model performance
  * execution time, memory consumption
  * Profile multiple steps.
* Auto profile and advise.
  * accelerator utilization check
  * expensive operation check
  * operation configuration check
  * distributed runtime check (Not OSS)

###Interfaces

* Python API
* Command Line
* Visualization
* C++ API (Not public, contact us if needed.)
