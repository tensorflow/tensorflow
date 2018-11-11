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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TESTS_GPU_CODEGEN_TEST_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TESTS_GPU_CODEGEN_TEST_H_

#include <string>

#include "tensorflow/compiler/xla/tests/llvm_irgen_test_base.h"

namespace xla {
namespace gpu {

// Tests that verify IR or PTX emitted by the GPU backend is as expected.
class GpuCodegenTest : public LlvmIrGenTestBase {
 protected:
  // Like HloTestBase::CreateNewModule(), with a flag for configuring the ftz
  // option.
  std::unique_ptr<HloModule> CreateNewModuleWithFTZ(bool ftz);

  // Compiles the given HLO module to PTX and verifies the PTX matches the given
  // FileCheck pattern.  (See http://llvm.org/docs/CommandGuide/FileCheck.html).
  void CompileAndVerifyPtx(std::unique_ptr<HloModule> hlo_module,
                           const string& pattern);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TESTS_GPU_CODEGEN_TEST_H_
