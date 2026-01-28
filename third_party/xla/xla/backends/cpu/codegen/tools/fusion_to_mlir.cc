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

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/tools/test_lib.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/init_main.h"

namespace xla::cpu {

absl::Status Run(const std::string& filename) {
  auto mlir_context = FusionCompiler::CreateContext();
  TF_ASSIGN_OR_RETURN(auto module, LoadTestModule(filename));
  auto* inst = module->entry_computation()->root_instruction();
  while (inst && (inst->opcode() == HloOpcode::kTuple ||
                  inst->opcode() == HloOpcode::kGetTupleElement)) {
    inst = inst->mutable_operand(0);
  }
  auto fusion = DynCast<HloFusionInstruction>(inst);
  fusion->SetAndSanitizeName("main");
  TF_ASSIGN_OR_RETURN(
      KernelDefinition kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, nullptr, false, false));
  llvm::outs() << kernel_definition.source().ToString();
  return absl::OkStatus();
}

}  // namespace xla::cpu

int main(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);
  CHECK_EQ(argc, 2) << "Must specify an input file";
  CHECK_OK(xla::cpu::Run(argv[1]));
  return 0;
}
