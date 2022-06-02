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

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/llvm_ir/alias_analysis.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuAlignmentTest : public GpuCodegenTest {};

TEST_F(GpuAlignmentTest, Test) {
  const char* hlo_string = R"(
HloModule GpuAlignmentTest

ENTRY main {
  zero = f32[] constant(0)
  tok = token[] after-all()
  a = f32[100] parameter(0)
  b_tup = (f32[200], token[]) infeed(tok)
  b = f32[200] get-tuple-element(b_tup), index=0
  a_padded = f32[150] pad(a, zero), padding=0_50
  b_sliced = f32[150] slice(b), slice={[0:150]}
  ROOT c = f32[150] add(a_padded, b_sliced)
}
)";

  CompileAndVerifyIr(hlo_string, R"(
CHECK: @fusion(ptr noalias align 16 dereferenceable(400) %alloc0, ptr noalias align 128 dereferenceable(600) %alloc1, ptr noalias align 128 dereferenceable(928) %temp_buf)
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
