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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

// Check that the ftz (flush denormals to zero) flag is reflected in PTX as
// expected.

namespace xla {
namespace gpu {
namespace {

class GpuFtzTest : public GpuCodegenTest {
 public:
  explicit GpuFtzTest(bool ftz) : ftz_(ftz) {}

  // Creates an HLO module that performs the given binary operation on some
  // data.
  std::unique_ptr<VerifiedHloModule> CreateBinaryOpModule(HloOpcode op) {
    HloComputation::Builder builder(TestName());

    Shape param_shape = ShapeUtil::MakeShapeWithLayout(
        F32, /*dimensions=*/{100, 100}, /*minor_to_major=*/{1, 0});
    HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
        /* parameter_number=*/0, param_shape, "x"));
    HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
        /* parameter_number=*/1, param_shape, "y"));
    builder.AddInstruction(HloInstruction::CreateBinary(param_shape, op, x, y));

    auto hlo_module = CreateNewVerifiedModuleWithFTZ(ftz_);
    hlo_module->AddEntryComputation(builder.Build());
    return hlo_module;
  }

  // Creates an HLO module that performs the given unary operation on some data.
  std::unique_ptr<VerifiedHloModule> CreateUnaryOpModule(HloOpcode op) {
    HloComputation::Builder builder(TestName());

    Shape param_shape = ShapeUtil::MakeShapeWithLayout(
        F32, /*dimensions=*/{100, 100}, /*minor_to_major=*/{1, 0});
    HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
        /* parameter_number=*/0, param_shape, "x"));
    builder.AddInstruction(HloInstruction::CreateUnary(param_shape, op, x));

    auto hlo_module = CreateNewVerifiedModuleWithFTZ(ftz_);
    hlo_module->AddEntryComputation(builder.Build());
    return hlo_module;
  }

  bool ftz_;
};

class GpuFtzEnabledTest : public GpuFtzTest {
 public:
  GpuFtzEnabledTest() : GpuFtzTest(/*ftz=*/true) {}
};

class GpuFtzDisabledTest : public GpuFtzTest {
 public:
  GpuFtzDisabledTest() : GpuFtzTest(/*ftz=*/false) {}
};

// Check that we emit mul.ftz.f32 when in ftz mode, and plain mul.f32 otherwise.
TEST_F(GpuFtzEnabledTest, MultiplyFtz) {
  CompileAndOptionallyVerifyPtx(CreateBinaryOpModule(HloOpcode::kMultiply), R"(
    CHECK-NOT: mul.rn.f32
    CHECK: mul.rn.ftz.f32
    CHECK-NOT: mul.rn.f32
  )");
}
TEST_F(GpuFtzDisabledTest, MultiplyFtz) {
  CompileAndOptionallyVerifyPtx(CreateBinaryOpModule(HloOpcode::kMultiply), R"(
    CHECK-NOT: mul.rn.ftz.f32
    CHECK: mul.rn.f32
    CHECK-NOT: mul.rn.ftz.f32
  )");
}

// In NVPTX, exp(float) is implemented in libdevice, and consults __nvvm_reflect
// to determine whether or not ftz is enabled.
// The implementation in CUDA 11 uses one ex2.approx.ftz, irrespective of ftz
// being enabled or not. In previous CUDA versions, there is a leading
// ex2.approx that does obey the ftz setting.
// Instead of pattern matching implementation details, it might be better to
// value-test the actual result instead. TODO(csigg): change to value-test.
TEST_F(GpuFtzEnabledTest, ExpFtz) {
  CompileAndOptionallyVerifyPtx(CreateUnaryOpModule(HloOpcode::kExp), R"(
    CHECK-NOT: ex2.approx.f32
    CHECK:     ex2.approx.ftz.f32
    CHECK-NOT: ex2.approx.f32
  )");
}

TEST_F(GpuFtzDisabledTest, ExpFtz) {
  CompileAndOptionallyVerifyPtx(CreateUnaryOpModule(HloOpcode::kExp), R"(
    CHECK:     ex2.approx.ftz.f32
    CHECK-NOT: ex2.approx.f32
    CHECK-NOT: ex2.approx.ftz.f32
  )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
