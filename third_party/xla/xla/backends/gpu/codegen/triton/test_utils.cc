/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/test_utils.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "google/protobuf/descriptor.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/instruction_fusion.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

std::vector<xla::PrimitiveType> AllXlaDataTypes() {
  std::vector<xla::PrimitiveType> xla_data_types;
  std::vector<xla::PrimitiveType> to_filter_out = {
      PRIMITIVE_TYPE_INVALID, TUPLE, BUFFER, OPAQUE_TYPE, TOKEN};
  const tsl::protobuf::EnumDescriptor* xla_type_descriptor =
      tsl::protobuf::GetEnumDescriptor<xla::PrimitiveType>();
  for (int enum_ix = 0; enum_ix < xla_type_descriptor->value_count();
       ++enum_ix) {
    xla::PrimitiveType xla_type = static_cast<xla::PrimitiveType>(
        xla_type_descriptor->value(enum_ix)->number());
    if (!absl::c_linear_search(to_filter_out, xla_type)) {
      xla_data_types.push_back(xla_type);
    }
  }
  return xla_data_types;
}

bool SupportsBF16(const stream_executor::GpuComputeCapability& cc) {
  if (cc.IsCuda()) {
    return cc.cuda_compute_capability()->IsAtLeast(
        se::CudaComputeCapability::kAmpere);
  }
  if (cc.IsRocm()) {
    return cc.rocm_compute_capability()->has_bf16_dtype_support();
  }
  CHECK(false);
}

absl::Status CreateTritonIrAndFileCheck(const HloModule* hlo_module,
                                        absl::string_view triton_fusion_name,
                                        absl::string_view filecheck_pattern) {
  auto* comp = hlo_module->GetComputationWithName(triton_fusion_name);
  TF_RET_CHECK(comp != nullptr) << absl::StrCat(
      "Computation '", triton_fusion_name, "' is not found in the module");
  auto fusion_backend_config = comp->FusionInstruction()
                                   ->backend_config<GpuBackendConfig>()
                                   ->fusion_backend_config();
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          fusion_backend_config.block_level_fusion_config());
  return CreateTritonIrAndFileCheck(*comp, block_level_parameters,
                                    filecheck_pattern);
}

absl::Status CreateTritonIrAndFileCheck(
    const HloComputation& computation,
    const BlockLevelParameters& block_level_parameters,
    absl::string_view filecheck_pattern) {
  auto* fusion = Cast<HloFusionInstruction>(computation.FusionInstruction());

  mlir::MLIRContext mlir_context;
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> triton_module,
      CreateTritonModule("triton_fn", fusion,
                         TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                         block_level_parameters, mlir_context));

  std::string out;
  llvm::raw_string_ostream os(out);
  triton_module->print(os);
  TF_ASSIGN_OR_RETURN(bool succeeded, RunFileCheck(out, filecheck_pattern));
  if (!succeeded) {
    return absl::InternalError("FileCheck failed.");
  }
  return absl::OkStatus();
}

absl::Status CreateTritonIrAndFileCheckForDot(
    const HloModule* hlo_module, absl::string_view triton_fusion_name,
    absl::string_view filecheck_pattern) {
  auto* comp = hlo_module->GetComputationWithName(triton_fusion_name);
  TF_RET_CHECK(comp != nullptr);
  return CreateTritonIrAndFileCheckForDot(*comp, filecheck_pattern);
}

absl::Status CreateTritonIrAndFileCheckForDot(
    const HloComputation& computation, absl::string_view filecheck_pattern) {
  BlockLevelParameters block_level_parameters;
  if (auto gpu_config =
          computation.FusionInstruction()->backend_config<GpuBackendConfig>();
      gpu_config.ok() && gpu_config->has_fusion_backend_config() &&
      gpu_config->fusion_backend_config().has_block_level_fusion_config()) {
    block_level_parameters = BlockLevelParameters::FromBlockLevelFusionConfig(
        gpu_config->fusion_backend_config().block_level_fusion_config());
  }
  return CreateTritonIrAndFileCheck(computation, block_level_parameters,
                                    filecheck_pattern);
}

absl::StatusOr<bool> ApplyFloatNormalization(
    HloModule* module, const stream_executor::GpuComputeCapability& cc) {
  const GpuFloatSupport bf16_support(cc, BF16);
  HloPassPipeline pipeline("hlo float normalization");
  pipeline.AddPass<FloatNormalization>(&bf16_support);
  return pipeline.Run(module);
}

}  // namespace xla::gpu
