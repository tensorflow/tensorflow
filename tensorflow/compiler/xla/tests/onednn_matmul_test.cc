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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <utility>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/tsl/platform/cpu_info.h"

namespace xla {
namespace cpu {

class MatmulTest : public HloTestBase {};

TEST_F(MatmulTest, SimpleTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32, entry_computation_layout={(f32[2,8,4,16]{3,2,1,0},f32[2,8,16,32]{3,2,1,0})->f32[2,8,4,32]{3,2,1,0}}

  ENTRY matmul.test.f32 {
    arg.0 = f32[2,8,4,16]{3,2,1,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,8,16,32]{3,2,1,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[2,8,4,32]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestBF16) {
  // TODO(penporn): Refactor IsBF16SupportedByOneDNNOnThisCPU() from
  // tensorflow/core/graph/mkl_graph_util.h and call the function instead.
  using tsl::port::TestCPUFeature;
  if (!TestCPUFeature(tsl::port::CPUFeature::AVX512_BF16) &&
      !TestCPUFeature(tsl::port::CPUFeature::AMX_BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* matmul_module_str = R"(
  HloModule matmul.test.bf16, entry_computation_layout={(bf16[2,8,4,16]{3,2,1,0},bf16[2,8,16,32]{3,2,1,0})->bf16[2,8,4,32]{3,2,1,0}}

  ENTRY matmul.test.bf16 {
    arg.0 = bf16[2,8,4,16]{3,2,1,0} parameter(0), parameter_replication={false}
    arg.1 = bf16[2,8,16,32]{3,2,1,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = bf16[2,8,4,32]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestF32TransposeB) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.1, entry_computation_layout={(f32[2,8,4,16]{3,1,2,0},f32[2,8,4,16]{3,1,2,0})->f32[2,8,4,4]{3,2,1,0}}

  ENTRY matmul.test.1 {
    arg.0 = f32[2,8,4,16]{3,1,2,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,8,4,16]{3,1,2,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[2,8,4,4]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
