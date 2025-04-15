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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TEST_UTILS_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TEST_UTILS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

std::vector<xla::PrimitiveType> AllXlaDataTypes();

bool SupportsBF16(const stream_executor::GpuComputeCapability& cc);

std::string ComputeCapabilityToString(
    const stream_executor::GpuComputeCapability& cc);

absl::Status CreateTritonIrAndFileCheck(HloTestBase* test,
                                        absl::string_view hlo_text,
                                        absl::string_view triton_fusion_name,
                                        absl::string_view filecheck_pattern);

absl::Status CreateTritonIrAndFileCheck(
    const HloComputation& computation,
    const BlockLevelParameters& block_level_parameters,
    absl::string_view filecheck_pattern);

absl::Status CreateTritonIrAndFileCheckForDot(
    HloTestBase* test, absl::string_view hlo_text,
    absl::string_view triton_fusion_name, absl::string_view filecheck_pattern);

absl::Status CreateTritonIrAndFileCheckForDot(
    const HloComputation& computation, absl::string_view filecheck_pattern);

inline BlockLevelParameters FromOutputTileSizes(
    std::vector<std::vector<int64_t>> output_tile_sizes) {
  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = std::move(output_tile_sizes);
  return block_level_parameters;
}

absl::StatusOr<bool> ApplyFloatNormalization(
    HloModule* module, const stream_executor::GpuComputeCapability& cc);

class TritonSupportTestBase : public HloTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override;

  // An HLO module together with a reference to the instruction of interest
  // that's being tested. See ParseTemplateAndGetInstruction for more details.
  class TestedInstruction {
   public:
    // Returns the HLO module.
    std::unique_ptr<HloModule>& Module() { return module_; };

    // The fusion instruction that calls the `triton_computation`.
    const HloFusionInstruction& TritonFusion() {
      return *Cast<HloFusionInstruction>(
          module_->entry_computation()->root_instruction());
    }

    // Returns the `triton_computation`.
    const HloComputation& TritonComputation() { return *instruction_.parent(); }

    // Returns the instruction within the `triton_computation` that has the
    // opcode provided to ParseAndGetInstruction.
    const HloInstruction& Instruction() { return instruction_; }

   private:
    friend TritonSupportTestBase;

    TestedInstruction(std::unique_ptr<HloModule> module,
                      const HloInstruction& instruction)
        : module_(std::move(module)), instruction_(instruction) {};
    std::unique_ptr<HloModule> module_;
    const HloInstruction& instruction_;
  };

  // Parses the given HLO template and returns the instruction that matches the
  // given opcode.
  //
  // The provided template must contain a computation called
  // `triton_computation`. If the template contains parameters $0 and $1, they
  // will be replaced with the data type and opcode respectively.
  // If the template's entry computation does not have a root fusion
  // instruction, a new entry computation will be created. The new computation
  // will have a root fusion instruction that has the same parameters as the
  // `triton_computation` and contains a fusion instruction that calls the
  // `triton_computation` with the generic Triton emitter. Tests that need
  // the `__triton_gemm` backend kind should provide their own ENTRY
  // computation.
  //
  // TODO(b/393299275): remove `use_nested_gemm_fusions` once the migration is
  // complete.
  absl::StatusOr<TestedInstruction> ParseTemplateAndGetInstruction(
      absl::string_view hlo_template, xla::PrimitiveType data_type,
      xla::HloOpcode opcode, bool use_nested_gemm_fusions = false);

  llvm::LLVMContext llvm_ctx_;
  llvm::Module llvm_module_{"module", llvm_ctx_};
  mlir::MLIRContext mlir_context_;
  TritonGemmConfig config_{16, 32, 512, 1, 4, 8};
};

class TritonSupportTestBaseWithParam
    : public TritonSupportTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, HloOpcode>> {};

std::string TritonSupportTestTypeAndOpcodeToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, HloOpcode>>& data);

std::string TritonSupportTestTypeAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, se::GpuComputeCapability>>& data);

std::string TritonSupportTestTypeAndOpcodeAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, HloOpcode, se::GpuComputeCapability>>& data);

std::string TritonSupportTestTwoTypesAndDeviceToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, PrimitiveType,
                                              se::GpuComputeCapability>>& data);

std::string TritonSupportTestDeviceToString(
    const ::testing::TestParamInfo<se::GpuComputeCapability>& data);

std::string TritonSupportTestTypeToString(
    const ::testing::TestParamInfo<PrimitiveType>& data);
}  //  namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TEST_UTILS_H_
