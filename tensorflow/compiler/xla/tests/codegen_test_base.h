/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_CODEGEN_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_CODEGEN_TEST_BASE_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

// Tests that verify IR emitted by the CPU/GPU backend is as expected.
class CodegenTestBase : public HloTestBase {
 protected:
  CodegenTestBase() {}

  // Returns the embedded LLVM IR from the given executable. Codegen tests must
  // override this method, but execution tests do not have to because they do
  // not examine the embedded IR.
  virtual string GetIrFromExecutable(const Executable& executable) = 0;

  // Compiles the given HLO module to LLVM IR and verifies the IR matches the
  // given pattern. `pattern` is in the FileCheck pattern matching syntax
  // (http://llvm.org/docs/CommandGuide/FileCheck.html).
  void CompileAndVerifyIr(std::unique_ptr<HloModule> hlo_module,
                          const string& pattern);

 protected:
  // Compiles hlo_module to an executable, CHECK-failing if this fails.
  std::unique_ptr<Executable> CompileToExecutable(
      std::unique_ptr<HloModule> hlo_module);

  // Runs FileCheck with the given pattern over the given string and EXPECTs
  // that FileCheck succeeded in matching the input.
  void RunFileCheck(const string& input, const string& pattern);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_CODEGEN_TEST_BASE_H_
