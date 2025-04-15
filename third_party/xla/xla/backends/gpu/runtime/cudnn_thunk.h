/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_CUDNN_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_CUDNN_THUNK_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

// Wraps executable cuDNN graph objects.
class CuDnnThunk : public Thunk {
 public:
  CuDnnThunk(std::string fingerprint, ThunkInfo,
             absl::Span<const KernelArgument>,
             std::optional<int64_t> sdpa_dropout_seed = std::nullopt);
  CuDnnThunk(const CuDnnThunk&) = delete;
  CuDnnThunk& operator=(const CuDnnThunk&) = delete;
  ~CuDnnThunk() override = default;

  absl::Status Initialize(const InitializeParams&) override;
  absl::Status ExecuteOnStream(const ExecuteParams&) override;

  std::shared_ptr<se::dnn::LazyDnnGraph> graph() const { return graph_; }
  const std::vector<BufferAllocation::Slice>& arguments() const {
    return args_;
  }

 private:
  absl::once_flag once_flag_;
  std::string fingerprint_;
  std::shared_ptr<se::dnn::LazyDnnGraph> graph_;
  std::vector<BufferAllocation::Slice> args_;
  // Sdpa dropout seed
  std::optional<int64_t> sdpa_dropout_seed_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_CUDNN_THUNK_H_
