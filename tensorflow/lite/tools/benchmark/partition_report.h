/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_PARTITION_REPORT_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_PARTITION_REPORT_H_

#include <string>

#include "tensorflow/lite/core/interpreter.h"

namespace tflite {
namespace benchmark {

// Returns a human-readable description of how the primary subgraph of
// `interpreter` is split between delegate kernels and nodes that fell back to
// the CPU interpreter. The report contains:
//   1. One line per entry of the execution plan: either
//      DELEGATE(<delegate name>) with the number of original nodes it replaced,
//      or the builtin/custom op name of a fallback node.
//   2. A summary: number of delegate partitions, number of fallback nodes, and
//      the fallback nodes grouped into contiguous "islands" with an opcode
//      histogram per island.
//   3. Boundary traffic: for every delegate partition, the bytes of its input
//      tensors that are produced by fallback nodes (or are graph inputs) and
//      of its output tensors consumed by fallback nodes (or graph outputs).
//      Constant tensors are not counted. Tensors handed directly from one
//      delegate partition to another are reported separately as
//      delegate->delegate traffic. Dynamic tensors are counted separately
//      since their size is only known after Invoke().
//   4. A single machine-friendly line, e.g.
//      PARTITION_REPORT partitions=3 delegated_nodes=61 fallback_nodes=7
//      boundary_bytes=1234567 inter_partition_bytes=0
//      dynamic_boundary_tensors=0
//
// Call this after all delegates have been applied and AllocateTensors() has
// succeeded, so that tensor sizes are final.
std::string GeneratePartitionReport(const Interpreter& interpreter);

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_PARTITION_REPORT_H_
