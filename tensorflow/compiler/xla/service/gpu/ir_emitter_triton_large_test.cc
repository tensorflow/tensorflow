/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"

namespace xla {
namespace gpu {
namespace {

class CompareTest : public GpuCodegenTest {};

TEST_F(CompareTest, IndexUsing64Bits) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f16[65536,32800] parameter(0)
  arg1 = f16[32800,32] parameter(1)
  ROOT custom-call = f16[65536,32] custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
}
)";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  p0 = f16[65536,32800] parameter(0)
  p1 = f16[32800,32] parameter(1)
  ROOT dot = f16[65536,32] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[65536,32800] parameter(0)
  p1 = f16[32800,32] parameter(1)
  ROOT _ = f16[65536,32] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config="{\"block_m\":\"32\",\"block_n\":\"32\",\"block_k\":\"32\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"1\"}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-3, 1e-3},
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
