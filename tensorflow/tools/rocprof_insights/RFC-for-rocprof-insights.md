# RFC for rocprof insights

## Introduction

`rocprof`, `rocprofv2`, and `rocprofv3` (rocprofiler-sdk) are the profiling tools that can be used to collect AMD hardware performance data when running applications with ROCm/HIP. The collected timeline trace data, which are JSON format for `rocprof` and `rocprofv2` and `pftrace` format for `rocprofv3`, can be visualized via `https://ui.perfetto.dev/` to guide the loop of profiling, analysis and optimization. To gain deep insights into specific running kernels, API launch and memory copy, it is necessary to obtain more statistics about them.

rocprof insights, which is developed as a Python package, aims to provide the following functionalities:

1. Data loading, including loading CSV and JSON files saved from `rocprof v1/v2/v3`
2. Data analysis, providing:
   - Total running time
   - Number of calls (instances)
   - Average time
   - Median time
   - Min/max time
   - StdDev time
   
   For each:
   - Kernel
   - HIP/HSA API calls
   - H2D, D2H, D2D memcopy
   - Checking private/group segment size (scratch/local memory, register spillage, shared local memory)

3. Data visualization, providing:
   - Pie chart plot of latency (running time)
   - Histogram of latency
   - Bar plot for kernels, API calls, etc.

Other features we would like to explore are:

1. Can we overlay the latency on top of the original operators in the computational graph (latency per op/node)?
2. Can we trace the input/output values of every node in the computational graph for checking accuracy (mismatch) per node?
