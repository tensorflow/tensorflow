/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_TESTS_CPU_CODEGEN_TEST_H_
#define XLA_SERVICE_CPU_TESTS_CPU_CODEGEN_TEST_H_

#include "xla/tests/llvm_irgen_test_base.h"

namespace xla {
namespace cpu {

// Tests that verify IR emitted by the CPU backend is as expected.
class CpuCodegenTest : public LlvmIrGenTestBase {};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_TESTS_CPU_CODEGEN_TEST_H_
