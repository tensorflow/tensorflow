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

#include "xla/backends/gpu/autotuner/miopen.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using MIOpenBackendConfig = stream_executor::dnn::AlgorithmProto;

namespace {

// Replaces the instruction with a new instruction with the same name in the
// parent computation. The given instruction will be replaced by a tuple of the
// convolution result and the workspace size. A few following instructions will
// be added to the parent computation to extract the convolution result from the
// new tuple.
absl::Status ApplyConfigAndUpdateWorkspaceInOutputTuple(
    HloInstruction& instr, const MIOpenBackendConfig& config) {
  HloComputation* computation = instr.parent();
  std::vector<Shape> new_call_element_shapes;
  // Add the shapes of the outputs of the convolution.
  new_call_element_shapes.reserve(instr.shape().tuple_shapes().size() - 1);
  for (int i = 0; i < instr.shape().tuple_shapes().size() - 1; ++i) {
    new_call_element_shapes.emplace_back(instr.shape().tuple_shapes(i));
  }
  // The final element is the size of the workspace.
  int64_t workspace_size = config.workspace_size().value();
  new_call_element_shapes.emplace_back(
      ShapeUtil::MakeShape(U8, {workspace_size}));
  Shape new_call_shape = ShapeUtil::MakeTupleShape(new_call_element_shapes);
  HloInstruction* new_call = computation->AddInstruction(
      instr.CloneWithNewOperands(new_call_shape, instr.operands()));
  new_call->SetAndSanitizeName(instr.name());

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      instr.backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig* cudnn_conv_config =
      gpu_backend_config.mutable_cudnn_conv_backend_config();
  *cudnn_conv_config->mutable_algorithm() = config;
  TF_RETURN_IF_ERROR(new_call->set_backend_config(gpu_backend_config));

  std::vector<HloInstruction*> new_tuple_elements;
  new_tuple_elements.reserve(new_call->shape().tuple_shapes().size() - 1);
  for (int i = 0; i < new_call->shape().tuple_shapes().size() - 1; ++i) {
    new_tuple_elements.emplace_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_call->shape().tuple_shapes(i), new_call, i)));
  }
  new_tuple_elements.emplace_back(computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({}))));

  // Repackage new_call so it has the same shape as the original call, namely
  // (conv_result, u8[0]).
  HloInstruction* new_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(new_tuple_elements));

  TF_RETURN_IF_ERROR(instr.parent()->ReplaceInstruction(&instr, new_tuple));
  return absl::OkStatus();
}

absl::Status ApplyConfigToMIOpenCustomCall(HloInstruction& instr,
                                           const MIOpenBackendConfig& config) {
  if (config.has_workspace_size() && config.workspace_size().value() > 0) {
    return ApplyConfigAndUpdateWorkspaceInOutputTuple(instr, config);
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig* cudnn_conv_config =
      gpu_config.mutable_cudnn_conv_backend_config();
  *cudnn_conv_config->mutable_algorithm() = config;
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  return absl::OkStatus();
}

}  // namespace

bool MIOpenBackend::IsSupported(const HloInstruction& instr) {
  return IsCustomCallToDnnConvolution(instr);
}

absl::StatusOr<std::unique_ptr<BackendConfig>> MIOpenBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (IsSupported(instr)) {
    MIOpenBackendConfig config;
    config.set_algo_id(-1);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(config);
    return any;
  }
  return absl::InvalidArgumentError(
      "MIOpen backend doesn't support getting a default config for this "
      "instruction.");
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
MIOpenBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (IsSupported(instr)) {
    MIOpenBackendConfig config;
    config.set_algo_id(-1);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(config);
    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.push_back(std::move(any));
    return configs;
  }
  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::Status MIOpenBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  MIOpenBackendConfig algorithm_config;
  if (!config.UnpackTo(&algorithm_config)) {
    return absl::InvalidArgumentError(
        "Failed to unpack MIOpenBackendConfig from Any.");
  }
  if (IsSupported(instr)) {
    return ApplyConfigToMIOpenCustomCall(instr, algorithm_config);
  }
  return absl::InvalidArgumentError(
      "MIOpen backend doesn't support this instruction.");
}

}  // namespace gpu
}  // namespace xla
