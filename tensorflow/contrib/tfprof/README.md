# tfprof: TensorFlow Profiler and Beyond

# Full Document in tensorflow/tools/tfprof/README.md

Author: Xin Pan (xpan@google.com, github: panyx0718), Jon Shlens, Yao Zhang

Consultants: Jon Shlens, Pete Warden

###Major Features

1.  Measure model parameters, float operations, tensor shapes.
2.  Profile op execution times, requested memory size and device placement.
3.  Inspect checkpoint tensors' shapes and their values.
4.  Selectively group, filter, account and order ops.

####tfprof supports 3 views to organize TensorFlow model profiles

    *  code view: Stats are associated your Python codes and organized as call stacks.
    *  scope view: Stats are organized as name scope hierarchies.
    *  graph view: Stats are organized as Tensorflow Op graph.

####For each view, there are 3 ways to display outputs:

    *  stdout: Results are written to stdout.
    *  timeline: Visualized in chrome browser as time series.
    *  file: Results are dumped to file.
