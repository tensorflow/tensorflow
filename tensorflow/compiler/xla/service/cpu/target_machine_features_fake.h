/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_FAKE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_FAKE_H_

#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"

namespace xla {
namespace cpu {
// Delegates calls to minimum_alignment_for_allocation to a user provided
// std::function, crashes on all other methods.
//
// Primarily useful for testing.
class TargetMachineFeaturesWithFakeAlignmentLogic
    : public TargetMachineFeatures {
 public:
  explicit TargetMachineFeaturesWithFakeAlignmentLogic(
      std::function<int64(int64)> fake_alignment_logic)
      : fake_alignment_logic_(std::move(fake_alignment_logic)) {}

  int vectorization_factor_in_bytes() const override {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

  int vector_register_byte_size(const llvm::Function& function) const override {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

  int vector_register_num_elements(const llvm::Function& function,
                                   PrimitiveType type) const override {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

  int vector_register_count(const llvm::Function& function) const override {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

  int64 minimum_alignment_for_allocation(int64 size_bytes) const override {
    return fake_alignment_logic_(size_bytes);
  }

 private:
  std::function<int64(int64)> fake_alignment_logic_;
};
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_FAKE_H_
