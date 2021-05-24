/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUSTOM_CALL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUSTOM_CALL_THUNK_H_

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"

namespace xla {
namespace gpu {

// Thunk to run a GPU custom call.
//
// This thunk's `ExecuteOnStream` implementation executes a host function
// `call_target` which is expected to enqueue operations onto the GPU.
//
// For information about the calling convention, see xla/g3doc/custom_call.md
//
// Note that not all kCustomCall HLOs in XLA:GPU end up being run by this thunk.
// XLA itself creates kCustomCall instructions when lowering kConvolution HLOs
// into calls to cudnn.  These internally-created custom-calls are run using
// ConvolutionThunk, not CustomCallThunk.  There's no ambiguity because they
// have special call target names (e.g. "__cudnn$convForward") that only the
// compiler is allowed to create.
class CustomCallThunk : public Thunk {
 public:
  using OptionalSlice = ::absl::optional<BufferAllocation::Slice>;
  CustomCallThunk(ThunkInfo thunk_info, void* call_target,
                  std::vector<OptionalSlice> operands,
                  std::vector<OptionalSlice> results,
                  const std::string& opaque);

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  void* call_target_;
  const std::vector<OptionalSlice> operands_;
  const std::vector<OptionalSlice> results_;
  const std::string opaque_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUSTOM_CALL_THUNK_H_
