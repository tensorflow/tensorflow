/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SELECT_K_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_SELECT_K_THUNK_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/types.h"  // IWYU pragma: keep

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// SelectKThunk
//===----------------------------------------------------------------------===//

// SelectKThunk executes the select_k operation on the provided inputs
class SelectKThunk : public Thunk {
 public:
  // Constructor.
  // Parameters:
  //   thunk_info       - ThunkInfo contains profile annotation & thunk id.
  //   batch_size       - Number of batches in the input tensor.
  //   num_elements     - Number of elements in each batch.
  //   k                - Number of top elements to select.
  //   dtype            - Data type of elements (e.g., F32, BF16).
  //   kernel_arguments - Kernel arguments holding buffer slices for
  //                      inputs/outputs.
  SelectKThunk(ThunkInfo thunk_info, std::uint32_t batch_size,
               std::uint32_t num_elements, std::uint32_t k,
               xla::PrimitiveType dtype,
               const emitters::KernelArguments& kernel_arguments);

  std::string ToString(int indent) const override;

  // Executes the TopK operation on the given stream.
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const std::vector<BufferAllocation::Slice>& arguments() const {
    return args_;
  }

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  std::uint32_t batch_size_;
  std::uint32_t num_elements_;
  std::uint32_t k_;
  xla::PrimitiveType dtype_;

  // Buffer slices passed to the kernel as arguments.
  std::vector<BufferAllocation::Slice> args_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SELECT_K_THUNK_H_
