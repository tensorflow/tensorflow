/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_SUPPORT_TEST_BASE_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_SUPPORT_TEST_BASE_H_

#include <memory>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

class SupportTestBase {
 protected:
  explicit SupportTestBase(
      std::function<
          absl::StatusOr<std::unique_ptr<VerifiedHloModule>>(absl::string_view)>
          parse_module_callback)
      : parse_module_callback_(parse_module_callback) {}
  // DebugOptions GetDebugOptionsForTest() const override;

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
    friend SupportTestBase;

    TestedInstruction(std::unique_ptr<HloModule> module,
                      const HloInstruction& instruction)
        : module_(std::move(module)), instruction_(instruction) {};
    std::unique_ptr<HloModule> module_;
    const HloInstruction& instruction_;
  };

  // Parses the given HLO template and returns the instruction that matches
  // the given opcode.
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
  absl::StatusOr<TestedInstruction> ParseTemplateAndGetInstruction(
      absl::string_view hlo_template, xla::PrimitiveType data_type,
      xla::HloOpcode opcode);

  llvm::LLVMContext llvm_ctx_;
  llvm::Triple target_triple_;
  std::string data_layout_;
  mlir::MLIRContext mlir_context_;
  TritonGemmConfig config_{16, 32, 512, 1, 4, 8};
  std::function<absl::StatusOr<std::unique_ptr<VerifiedHloModule>>(
      absl::string_view)>
      parse_module_callback_;
};

std::string ComputeCapabilityToString(
    const stream_executor::GpuComputeCapability& cc);

std::string SupportTestTypeAndOpcodeToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, HloOpcode>>& data);

std::string SupportTestTypeAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, se::GpuComputeCapability>>& data);

std::string SupportTestTypeAndOpcodeAndDeviceToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, HloOpcode, se::GpuComputeCapability>>& data);

std::string SupportTestTwoTypesAndDeviceToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, PrimitiveType,
                                              se::GpuComputeCapability>>& data);

std::string SupportTestDeviceToString(
    const ::testing::TestParamInfo<se::GpuComputeCapability>& data);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_SUPPORT_TEST_BASE_H_
