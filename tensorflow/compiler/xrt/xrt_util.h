/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Utility functions in support of the XRT API.

#ifndef TENSORFLOW_COMPILER_XRT_XRT_UTIL_H_
#define TENSORFLOW_COMPILER_XRT_XRT_UTIL_H_

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_memory_manager.h"
#include "tensorflow/compiler/xrt/xrt_refptr.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

struct InputCoords {
  explicit InputCoords(int64 handle) : handle(handle) {}
  InputCoords(int64 handle, xla::ShapeIndex index)
      : handle(handle), index(std::move(index)) {}

  int64 handle = 0;
  xla::ShapeIndex index;
};

// Filters the debug options provided as argument according to the value of the
// TF_XLA_DEBUG_OPTIONS_PASSTHROUGH environment variable. If such variable is
// set to "1" or "true", the debug options will be returned as is. Otherwise
// only a subset of them will be set in the returned ones, and all the paths
// contained in it, will be limited to gs:// and bigstore:// ones.
xla::DebugOptions BuildXlaDebugOptions(const xla::DebugOptions& ref_options);

// Populates the input_coords with a list of input coordinates from a input_name
// op argument.
xla::StatusOr<std::vector<InputCoords>> GetComputationInputs(
    OpKernelContext* context, const char* input_name);

// Create the XRT execute output tensor given the computation result
// (output_tuple). The return_exploded_tuple tells whether a tuple result should
// be returned as vector of handles representing each tuple child.
Status CreateExecuteOutput(OpKernelContext* context,
                           XRTMemoryManager* memory_manager,
                           RefPtr<XRTTupleAllocation> output_tuple,
                           bool return_exploded_tuple);

// Drives the XRT chained computation execution given the supplied core execute
// function.
using ChainedExecuteFn =
    std::function<xla::StatusOr<RefPtr<XRTTupleAllocation>>(
        const xrt::XRTChainedExecuteOp&,
        absl::Span<const RefPtr<XRTTupleAllocation>>)>;
Status ExecuteChained(OpKernelContext* context,
                      const RefPtr<XRTMemoryManager>& memory_manager,
                      xla::Backend* backend, int device_ordinal,
                      const xrt::XRTChainedExecutePlan& plan,
                      const xrt::XRTChainedExecuteConfig& config,
                      const ChainedExecuteFn& execute_op);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_UTIL_H_
