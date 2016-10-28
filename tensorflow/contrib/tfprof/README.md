# tfprof: A Profiling Tool for TensorFlow Models

Internal User Please Use: go/tfprof

Author: Xin Pan (xpan@google.com, github: panyx0718)

Consultants: Jon Shlens, Pete Warden


## Introduction

tfprof is a profiling tool for TensorFlow that analyzes model architectures
and measures system performance.

###Major Features

1.  Measure model parameters, float operations, tensor shapes.
2.  Measure op execution times, requested memory size and device placement.
3.  Inspect checkpoint tensors' shapes and their values.
4.  Explore model based on name scope or graph structure.
5.  Selectively grouping/filtering/accounting/ordering ops.

tfprof can be used as CommandLine Interface (CLI) and Python API.
CLI locates in tensorflow/tools/tfprof.
Python API locates in tensorflow/contrib/tfprof.
Tutorial locates in tensorflow/tools/tfprof/README.md

Enjoy!