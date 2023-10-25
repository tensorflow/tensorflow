/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_LIGHT_OUTSIDE_COMPILATION_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_LIGHT_OUTSIDE_COMPILATION_H_

#include <map>

#include "tensorflow/compiler/tf2xla/kernels/callback.pb.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Using std::map as the maps are presumed to be tiny, and we want a
// deterministic iteration order.
//
// Dimension -> bound.
using DimensionBoundsMap = std::map<int, int>;

// Output -> dimension -> bound.
using OutputDimensionBoundsMap = std::map<int, DimensionBoundsMap>;

// Generic kernel for registering TF2XLA kernels which call back into the TF
// runtime to run a given kernel defined by the wrapped node.
//
// Cf. example usages in light_outside_compilation_kernels_for_test.cc.
//
// Currently does not support dynamic shape or resource variables. Currently
// works only on GPU.
class LightOutsideCompilationOp : public XlaOpKernel {
 public:
  explicit LightOutsideCompilationOp(OpKernelConstruction* context);
  void Compile(XlaOpKernelContext* ctx) override;

  // Override to provide statically known bounds on output in case of dynamic
  // shapes.
  virtual StatusOr<OutputDimensionBoundsMap> DynamicOutputDimensions(
      const NodeDef& ndef, XlaOpKernelContext* ctx) const {
    return OutputDimensionBoundsMap{};
  }

 private:
  Status CompileToCustomCallCallingTfKernel(int graph_def_version,
                                            const NodeDef& node_def,
                                            XlaOpKernelContext* ctx);
  static Status CallTfKernel(void* stream_handle, void** buffers,
                             const char* opaque, int opaque_len);

  NodeDef def_;
  int graph_def_version_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_LIGHT_OUTSIDE_COMPILATION_H_
