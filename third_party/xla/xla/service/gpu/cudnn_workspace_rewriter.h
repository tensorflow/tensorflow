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

#ifndef XLA_SERVICE_GPU_CUDNN_WORKSPACE_REWRITER_H_
#define XLA_SERVICE_GPU_CUDNN_WORKSPACE_REWRITER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Rewrite cuDNN custom call to have correct workspace size by build graph
// and serialize so we can use it later
class CuDnnWorkspaceRewriter : public HloModulePass {
 public:
  // <HLO computation fingerprint, serialized compiled cuDNN graph>.
  using BinaryMap = absl::flat_hash_map<std::string, std::string>;

  explicit CuDnnWorkspaceRewriter(se::StreamExecutor& stream_exec)
      : dnn_support_(*stream_exec.AsDnn()) {}

  absl::string_view name() const override { return "cudnn-workspace-rewriter"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::dnn::DnnSupport& dnn_support_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUDNN_WORKSPACE_REWRITER_H_
