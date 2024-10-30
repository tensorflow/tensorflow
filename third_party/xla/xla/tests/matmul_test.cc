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

#include <utility>

#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

class MatmulTestWithCublas : public HloTestBase,
                             public ::testing::WithParamInterface<bool> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(use_cublas_lt_);
    return debug_options;
  }
  void SetUp() override {
    auto dbg = GetDebugOptionsForTest();
    if (dbg.xla_gpu_enable_cublaslt()) {
      const auto& gpu_cc = backend()
                               .default_stream_executor()
                               ->GetDeviceDescription()
                               .gpu_compute_capability();
      if (auto* rocm = std::get_if<se::RocmComputeCapability>(&gpu_cc);
          rocm != nullptr && !rocm->has_hipblaslt()) {
        GTEST_SKIP() << "No hipblas-lt support on this architecture!";
      }
    }
  }

 private:
  const bool use_cublas_lt_{GetParam()};
};

TEST_P(MatmulTestWithCublas, GemmRewriter_RegressionTestF64) {
  const char* module_str = R"(
HloModule GeneralMatMulActivation.7, entry_computation_layout={(f64[2,2,2]{2,1,0}, f64[2,2,2]{2,1,0})->f64[2,2,2]{2,1,0}}

ENTRY GeneralMatMulActivation.7 {
  x.1 = f64[2,2,2]{2,1,0} parameter(0)
  y.2 = f64[2,2,2]{2,1,0} parameter(1)
  dot.3 = f64[2,2,2]{2,1,0} dot(x.1, y.2), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  constant.4 = f64[] constant(0)
  broadcast.5 = f64[2,2,2]{2,1,0} broadcast(constant.4), dimensions={}
  ROOT maximum.6 = f64[2,2,2]{2,1,0} maximum(dot.3, broadcast.5)
})";

  EXPECT_TRUE(RunAndCompare(module_str, ErrorSpec{1e-4, 1e-4}));
}

// There was an issue where the compilation process of an Inverse operation was
// resulting in a cached cuBLASLt matmul plan which was incorrectly fetched at
// the time of the Matmul operation
TEST_P(MatmulTestWithCublas, InverseAndMatmul) {
  const char* inverse_module_str = R"(
  HloModule MatrixInverse.26, entry_computation_layout={(f32[2,6,2,2]{3,2,1,0})->f32[2,6,2,2]{3,2,1,0}}

  ENTRY MatrixInverse.26 {
    iota.10 = s32[2]{0} iota(), iota_dimension=0
    constant.11 = s32[] constant(-1)
    broadcast.12 = s32[2]{0} broadcast(constant.11), dimensions={}
    add.13 = s32[2]{0} add(iota.10, broadcast.12)
    broadcast.15 = s32[2,2]{1,0} broadcast(add.13), dimensions={0}
    iota.9 = s32[2]{0} iota(), iota_dimension=0
    broadcast.14 = s32[2,2]{1,0} broadcast(iota.9), dimensions={1}
    compare.16 = pred[2,2]{1,0} compare(broadcast.15, broadcast.14), direction=GE
    broadcast.17 = pred[2,6,2,2]{3,2,1,0} broadcast(pred[2,2]{1,0} compare.16), dimensions={2,3}
    constant.18 = f32[] constant(0)
    broadcast.19 = f32[2,6,2,2]{3,2,1,0} broadcast(constant.18), dimensions={}
    arg0.1 = f32[2,6,2,2]{3,2,1,0} parameter(0), parameter_replication={false}
    reshape.2 = f32[2,6,2,2]{3,2,1,0} reshape(arg0.1)
    custom-call.3 = (f32[2,6,2,2]{3,2,1,0}, f32[2,6,2]{2,1,0}) custom-call(reshape.2), custom_call_target="Qr"
    get-tuple-element.4 = f32[2,6,2,2]{3,2,1,0} get-tuple-element(custom-call.3), index=0
    slice.8 = f32[2,6,2,2]{3,2,1,0} slice(get-tuple-element.4), slice={[0:2], [0:6], [0:2], [0:2]}
    select.20 = f32[2,6,2,2]{3,2,1,0} select(broadcast.17, broadcast.19, slice.8)
    get-tuple-element.5 = f32[2,6,2]{2,1,0} get-tuple-element(custom-call.3), index=1
    custom-call.6 = f32[2,6,2,2]{3,2,1,0} custom-call(get-tuple-element.4, get-tuple-element.5), custom_call_target="ProductOfElementaryHouseholderReflectors"
    slice.7 = f32[2,6,2,2]{3,2,1,0} slice(custom-call.6), slice={[0:2], [0:6], [0:2], [0:2]}
    transpose.21 = f32[2,6,2,2]{2,3,1,0} transpose(slice.7), dimensions={0,1,3,2}
    triangular-solve.22 = f32[2,6,2,2]{2,3,1,0} triangular-solve(select.20, transpose.21), left_side=true, transpose_a=NO_TRANSPOSE
    reshape.23 = f32[2,6,2,2]{3,2,1,0} reshape(triangular-solve.22)
    tuple.24 = (f32[2,6,2,2]{3,2,1,0}) tuple(reshape.23)
    ROOT get-tuple-element.25 = f32[2,6,2,2]{3,2,1,0} get-tuple-element(tuple.24), index=0
  })";

  const char* matmul_module_str = R"(
  HloModule MatMul.10, entry_computation_layout={(f32[2,6,2,2]{3,2,1,0},f32[2,6,2,2]{3,2,1,0})->f32[2,6,2,2]{3,2,1,0}}

  ENTRY MatMul.10 {
    arg0.1 = f32[2,6,2,2]{3,2,1,0} parameter(0), parameter_replication={false}
    reshape.3 = f32[2,6,2,2]{3,2,1,0} reshape(arg0.1)
    arg1.2 = f32[2,6,2,2]{3,2,1,0} parameter(1), parameter_replication={false}
    reshape.4 = f32[2,6,2,2]{3,2,1,0} reshape(arg1.2)
    dot.5 = f32[2,6,2,2]{3,2,1,0} dot(reshape.3, reshape.4), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    transpose.6 = f32[2,6,2,2]{3,2,1,0} transpose(dot.5), dimensions={0,1,2,3}
    reshape.7 = f32[2,6,2,2]{3,2,1,0} reshape(transpose.6)
    tuple.8 = (f32[2,6,2,2]{3,2,1,0}) tuple(reshape.7)
    ROOT get-tuple-element.9 = f32[2,6,2,2]{3,2,1,0} get-tuple-element(tuple.8), index=0
  })";

  EXPECT_TRUE(RunAndCompare(inverse_module_str, ErrorSpec{1e-4, 1e-4}));
  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

INSTANTIATE_TEST_SUITE_P(UsingCublasLt, MatmulTestWithCublas,
                         ::testing::Bool());

}  // namespace
}  // namespace xla
