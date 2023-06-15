/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_compiler.h"

#include <memory>

#include "absl/status/status.h"
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_compiler.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace {

bool IsGpuClient(const PjRtClient& client) {
  return client.platform_id() == GpuId();
}

bool IsSameTopology(const PjRtTopologyDescription& topology1,
                    const PjRtTopologyDescription& topology2) {
  const StreamExecutorGpuTopologyDescription& gpu_topology1 =
      tensorflow::down_cast<const StreamExecutorGpuTopologyDescription&>(
          topology1);
  const StreamExecutorGpuTopologyDescription& gpu_topology2 =
      tensorflow::down_cast<const StreamExecutorGpuTopologyDescription&>(
          topology2);
  return gpu_topology1 == gpu_topology2;
}

absl::Status IsValidTopologyAndClientForCompile(
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  if (client == nullptr) {
    return absl::UnimplementedError(
        "SE:GPU compiler requires non-null client.");
  }
  if (!IsGpuClient(*client)) {
    return absl::InvalidArgumentError(
        "SE:GPU compiler requires a GPU PjRtClient.");
  }
  TF_ASSIGN_OR_RETURN(auto client_topology, client->GetTopologyDescription());

  if (!IsSameTopology(topology, *client_topology)) {
    return absl::UnimplementedError(
        "SE:GPU compiler requires the topology same as the one in the client.");
  }
  return absl::OkStatus();
}
}  // namespace

// TODO(b/285385306): Enable compilation on provided `topology`.
absl::StatusOr<std::unique_ptr<PjRtExecutable>>
StreamExecutorGpuCompiler::Compile(CompileOptions options,
                                   const XlaComputation& computation,
                                   const PjRtTopologyDescription& topology,
                                   PjRtClient* client) {
  TF_RETURN_IF_ERROR(IsValidTopologyAndClientForCompile(topology, client));
  return client->Compile(computation, options);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
StreamExecutorGpuCompiler::Compile(CompileOptions options,
                                   mlir::ModuleOp module,
                                   const PjRtTopologyDescription& topology,
                                   PjRtClient* client) {
  TF_RETURN_IF_ERROR(IsValidTopologyAndClientForCompile(topology, client));
  return client->Compile(module, options);
}

REGISTER_MODULE_INITIALIZER(pjrt_register_se_gpu_compiler, {
  std::unique_ptr<PjRtCompiler> compiler =
      std::make_unique<StreamExecutorGpuCompiler>();
  PjRtRegisterCompiler(GpuName(), std::move(compiler));
});
}  // namespace xla
