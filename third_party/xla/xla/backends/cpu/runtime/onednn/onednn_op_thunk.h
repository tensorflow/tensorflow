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

#ifndef XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_OP_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_OP_THUNK_H_

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

class OneDnnOpThunk : public Thunk {
 public:
  ~OneDnnOpThunk() override;

  // Buffer allocation slices and shapes.
  struct OpBuffers {
    std::vector<BufferAllocation::Slice> arguments_buffers;
    std::vector<Shape> arguments_shapes;

    std::vector<BufferAllocation::Slice> results_buffers;
    std::vector<Shape> results_shapes;
    bool is_tuple_result;
  };

  // Variant config for supported oneDNN ops.
  // TODO(intel-tf): Add more oneDNN operation configs as needed.
  using OneDnnOpConfig =
      std::variant<OneDnnMatMulConfig, OneDnnConvolutionConfig,
                   OneDnnNormConfig, OneDnnSoftmaxConfig>;

  static absl::StatusOr<std::unique_ptr<OneDnnOpThunk>> Create(
      const std::string& custom_call_target, Info info, OpBuffers buffers,
      OneDnnOpConfig config);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

 private:
  OneDnnOpThunk(const std::string& custom_call_target, Info info,
                OpBuffers buffers, OneDnnOpConfig config);

  // oneDNN runtime instantiated for the oneDNN operation.
  struct OneDnnRuntime;

  OpBuffers op_buffers_;
  OneDnnOpConfig config_;
  std::string target_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_OP_THUNK_H_
