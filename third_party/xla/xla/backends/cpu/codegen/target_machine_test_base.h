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

#ifndef XLA_BACKENDS_CPU_CODEGEN_TARGET_MACHINE_TEST_BASE_H_
#define XLA_BACKENDS_CPU_CODEGEN_TARGET_MACHINE_TEST_BASE_H_

#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "llvm-c/Target.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"

namespace xla::cpu {

class TargetMachineTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeARMTarget();
    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTargetMC();
  }

  std::unique_ptr<llvm::TargetMachine> CreateTargetMachine(
      absl::string_view triple_string, absl::string_view cpu_name,
      absl::string_view features) {
    std::string error;
    const llvm::Target* target =
        llvm::TargetRegistry::lookupTarget(triple_string, error);
    if (target == nullptr) {
      LOG(ERROR) << "Failed to lookup target: " << error;
    }

    llvm::Triple triple(triple_string);
    llvm::TargetOptions target_options;
    return absl::WrapUnique(target->createTargetMachine(
        triple, cpu_name, features, target_options, /*RM=*/std::nullopt));
  }

  std::unique_ptr<TargetMachineFeatures> CreateTargetMachineFeatures(
      absl::string_view triple_string, absl::string_view cpu_name,
      absl::string_view features) {
    std::unique_ptr<llvm::TargetMachine> target_machine =
        CreateTargetMachine(triple_string, cpu_name, features);
    return std::make_unique<TargetMachineFeatures>(target_machine.get());
  }
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_TARGET_MACHINE_TEST_BASE_H_
