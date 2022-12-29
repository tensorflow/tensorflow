/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class ElementalIrEmitterTest : public GpuCodegenTest {
 protected:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_F(ElementalIrEmitterTest, TestConvertF32ToBF16) {
  const char* hlo_string = R"(
    HloModule convertF32ToBF16

    ENTRY main {
      f32_ = f32[] parameter(0)
      ROOT bf16_ = bf16[] convert(f32[] f32_)
    }
  )";

  if (GetCudaComputeCapability().IsAtLeast(8)) {
    CompileAndVerifyIr(hlo_string, R"(
CHECK: call i16 @llvm.nvvm.f2bf16.rn(float %{{.*}})
)");
  } else {
    CompileAndVerifyIr(hlo_string, R"(
CHECK-NOT: nvvm.f2bf16
)");

    CompileAndVerifyIr(hlo_string, R"(
CHECK: bitcast float %{{.*}} to i32
CHECK: trunc i32 %{{.*}} to i16
)");
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
