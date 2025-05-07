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

#include <memory>
#include <utility>

#include "absl/strings/str_replace.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {

class GpuCopyTest : public GpuCodegenTest {};

// The GPU backend should not emit a copy kernel for the kCopy instruction in
// this test. Instead, it should generate a CopyThunk which invokes cuMemcpy at
// runtime.
TEST_F(GpuCopyTest, UseMemcpy) {
  HloComputation::Builder builder(TestName());

  Literal literal = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  // There should not be any kernel prefixed "copy".
  CompileAndVerifyIr(std::move(hlo_module), "; CHECK-NOT: define void @_copy",
                     /*match_optimized_ir=*/false);
}

TEST_F(GpuCopyTest, CopyTranspose) {
  const char* hlo_text = R"(
    HloModule Test

    fused_computation {
      param_0 = f32[100,200,300]{2,1,0} parameter(0)
      ROOT b.1 = f32[100,200,300]{2,0,1} copy(f32[100,200,300]{2,1,0} param_0)
    }

    ENTRY main {
      a = f32[100, 200, 300]{2,1,0} parameter(0)
      ROOT wrapped_b = f32[100,200,300]{2,0,1} fusion(f32[100,200,300]{2,1,0} %a), kind=kLoop, calls=fused_computation
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

constexpr char kSliceMemcpyModule[] = R"(
    dynamic_slice {
      p0 = s32[4,8,8]{2,1,0} parameter(0)
      p1 = s32[] parameter(1)
      c1 = s32[] constant(1)
      p2 = s32[] parameter(2)

      p1p1 = s32[] add(p1, c1)

      // Test all supported kinds of offsets: derived from the while loop's
      // induction variable (p1p1), constant (c1) and always clamped to 0, so
      // the value is irrelevant (p2).
      ROOT slice = s32[1,1,8] dynamic-slice(p0, p1p1, c1, p2),
          dynamic_slice_sizes={1,1,8}
    }

    remainder {
      p0 = s32[] parameter(0)
      c5 = s32[] constant(5)
      // We take the value modulo 5 to test for correct clamping (the offset 4
      // must get clamped to 3, since it's greater or equal than the dimension
      // size).
      ROOT remainder = s32[] remainder(p0, c5)
    }

    add {
      p0 = s32[] parameter(0)
      c1 = s32[] constant(1)
      ROOT sum = s32[] add(p0, c1)
    }

    add_slices {
      p0 = s32[1,1,8] parameter(0)
      p1 = s32[1,1,8] parameter(1)
      ROOT sum = s32[1,1,8] add(p0, p1)
    }

    times_two {
      p0 = s32[] parameter(0)
      ROOT sum = s32[] add(p0, p0)
    }

    body {
      p0 = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8]{2,1,0} get-tuple-element(p0), index=1

      ivar_copy = s32[] copy(ivar)
      acc = s32[1,1,8] get-tuple-element(p0), index=2
      acc_copy = s32[1,1,8] copy(acc)

      offset1 = s32[] fusion(ivar_copy), kind=kLoop, calls=remainder
      offset2 = s32[] get-tuple-element(p0), index=3

      slice = s32[1,1,8] fusion(input, offset1, offset2), kind=kLoop, calls=dynamic_slice,
          backend_config={"fusion_backend_config":{"kind":"__dynamic_memcpy"}}
      next_ivar = s32[] fusion(ivar_copy), kind=kLoop, calls=add
      next_offset_2 = s32[] fusion(offset2), kind=kLoop, calls=times_two

      next_acc = s32[1,1,8] fusion(acc_copy, slice), kind=kLoop, calls=add_slices
      ROOT result = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[])
          tuple(next_ivar, input, next_acc, next_offset_2)
    }

    compare {
      p0 = s32[] parameter(0)
      c6 = s32[] constant(6)
      ROOT cmp = pred[] compare(p0, c6), direction=LT
    }

    condition {
      p0 = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      ROOT cmp = pred[] fusion(ivar), kind=kLoop, calls=compare
    }

    zero {
      c0 = s32[] constant(0)
      ROOT bc = s32[1,1,8] broadcast(c0), dimensions={}
    }

    input {
      iota = s32[256] iota(), iota_dimension=0
      ROOT bc = s32[4,8,8]{2,1,0} bitcast(iota)
    }

    ENTRY main {
      input = s32[4,8,8]{2,1,0} fusion(), kind=kLoop, calls=input
      init_acc = s32[1,1,8] fusion(), kind=kLoop, calls=zero
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      tuple = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[]) tuple(c0, input, init_acc, c1)
      ROOT while = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"6"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

TEST_F(GpuCopyTest, UseMemcpyForDynamicSlice) {
  // This verifies that dynamic slices can be implemented using memcpy in
  // certain conditions.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kSliceMemcpyModule));

  // There should not be a kernel for `dynamic_slice`.
  CompileAndVerifyIr(std::move(hlo_module), "; CHECK-NOT: void @slice",
                     /*match_optimized_ir=*/false,
                     /*run_optimization_passes=*/false);
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kSliceMemcpyModule, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(GpuCopyTest, DoNotUseMemcpyForDynamicSlice) {
  // This is a test for the CompileAndVerifyIr statement in
  // UseMemcpyForDynamicSlice. When the conditions are not met, there should be
  // a fusion for the slice.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kSliceMemcpyModule));

  // This prevents the memcpy fusion logic from triggering.
  hlo_module->entry_computation()->root_instruction()->clear_backend_config();

  CompileAndVerifyIr(std::move(hlo_module), "; CHECK: void @slice",
                     /*match_optimized_ir=*/false,
                     /*run_optimization_passes=*/false);
}

TEST_F(GpuCopyTest, DoNotUseMemcpyWithLayoutChange) {
  // By changing the layout of the result, the slice is no longer contiguous and
  // cannot be emitted with a memcpy.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                              kSliceMemcpyModule, {{"{2,1,0}", "{0,2,1}"}})));

  CompileAndVerifyIr(std::move(hlo_module), "; CHECK: void @slice",
                     /*match_optimized_ir=*/false,
                     /*run_optimization_passes=*/false);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kSliceMemcpyModule, ErrorSpec{0, 0}));
}

constexpr char kDynamicUpdateSliceModule[] = R"(
    dynamic_update_slice {
      p0 = s32[4,8,8] parameter(0)
      p1 = s32[1,1,8] parameter(1)
      p2 = s32[] parameter(2)
      c0 = s32[] constant(0)

      ROOT update-slice = s32[4,8,8] dynamic-update-slice(p0, p1, p2, c0, c0)
    }

    add {
      p0 = s32[] parameter(0)
      c1 = s32[] constant(1)
      ROOT sum = s32[] add(p0, c1)
    }

    add_slices {
      p0 = s32[1,1,8] parameter(0)
      ROOT sum = s32[1,1,8] add(p0, p0)
    }

    body {
      p0 = (s32[], s32[4,8,8], s32[1,1,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8] get-tuple-element(p0), index=1
      input-copy = s32[4,8,8] copy(input)

      ivar_copy = s32[] copy(ivar)
      acc = s32[1,1,8] get-tuple-element(p0), index=2
      acc_copy = s32[1,1,8] copy(acc)

      updated = s32[4,8,8] fusion(input-copy, acc_copy, ivar_copy), kind=kLoop,
          calls=dynamic_update_slice,
          backend_config={"fusion_backend_config":{"kind":"__dynamic_memcpy"}}
      next_ivar = s32[] fusion(ivar_copy), kind=kLoop, calls=add

      next_acc = s32[1,1,8] fusion(acc_copy), kind=kLoop, calls=add_slices
      ROOT result = (s32[], s32[4,8,8], s32[1,1,8])
          tuple(next_ivar, updated, next_acc)
    }

    compare {
      p0 = s32[] parameter(0)
      c6 = s32[] constant(6)
      ROOT cmp = pred[] compare(p0, c6), direction=LT
    }

    condition {
      p0 = (s32[], s32[4,8,8], s32[1,1,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      ROOT cmp = pred[] fusion(ivar), kind=kLoop, calls=compare
    }

    input {
      iota = s32[256] iota(), iota_dimension=0
      ROOT bc = s32[4,8,8] bitcast(iota)
    }

    ENTRY main {
      input = s32[4,8,8] fusion(), kind=kLoop, calls=input
      init_acc = s32[1,1,8] constant({{{7,6,5,4,3,2,1,0}}})
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8,8], s32[1,1,8]) tuple(c0, input, init_acc)
      ROOT while = (s32[], s32[4,8,8], s32[1,1,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"6"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

TEST_F(GpuCopyTest, UseMemcpyForDynamicUpdateSlice) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> hlo_module,
      ParseAndReturnVerifiedModule(kDynamicUpdateSliceModule));

  CompileAndVerifyIr(std::move(hlo_module), "; CHECK-NOT: void @updated",
                     /*match_optimized_ir=*/false,
                     /*run_optimization_passes=*/false);
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kDynamicUpdateSliceModule, ErrorSpec{0, 0}));
}

TEST_F(GpuCopyTest, DoNotUseMemcpyForDynamicUpdateSlice) {
  // This is a test for the CompileAndVerifyIr statement in
  // UseMemcpyForDynamicUpdateSlice. When the conditions are not met, there
  // should be a fusion for the slice.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> hlo_module,
      ParseAndReturnVerifiedModule(kDynamicUpdateSliceModule));

  // This prevents the memcpy fusion logic from triggering.
  hlo_module->entry_computation()->root_instruction()->clear_backend_config();
  CompileAndVerifyIr(std::move(hlo_module), "; CHECK: void @updated",
                     /*match_optimized_ir=*/false,
                     /*run_optimization_passes=*/false);
}

constexpr char kDynamicUpdateSliceWithBitcastModule[] = R"(
    dynamic_update_slice {
      p0 = s32[8,8] parameter(0)
      p1 = s32[1,4] parameter(1)
      p2 = s32[] parameter(2)
      bc0 = s32[64] bitcast(p0)
      bc1 = s32[4] bitcast(p1)
      update-slice = s32[64] dynamic-update-slice(bc0, bc1, p2)
      ROOT bc = s32[8,8] bitcast(update-slice)
    }

    add {
      p0 = s32[] parameter(0)
      c1 = s32[] constant(1)
      ROOT sum = s32[] add(p0, c1)
    }

    body {
      while_arg = (s32[], s32[8,8], s32[1,4]) parameter(0)
      ivar = s32[] get-tuple-element(while_arg), index=0
      input = s32[8,8] get-tuple-element(while_arg), index=1
      update = s32[1,4] get-tuple-element(while_arg), index=2
      input-copy = s32[8,8] copy(input)
      ivar-copy = s32[] copy(ivar)

      updated_bc = s32[8,8] fusion(input-copy, update, ivar-copy), kind=kLoop,
          calls=dynamic_update_slice,
          backend_config={"fusion_backend_config":{"kind":"__dynamic_memcpy"}}
      next_ivar = s32[] fusion(ivar-copy), kind=kLoop, calls=add

      ROOT result = (s32[], s32[8,8], s32[1,4])
          tuple(next_ivar, updated_bc, update)
    }

    compare {
      p0 = s32[] parameter(0)
      c6 = s32[] constant(6)
      ROOT cmp = pred[] compare(p0, c6), direction=LT
    }

    condition {
      while_arg = (s32[], s32[8,8], s32[1,4]) parameter(0)
      ivar = s32[] get-tuple-element(while_arg), index=0
      ROOT cmp = pred[] fusion(ivar), kind=kLoop, calls=compare
    }

    input {
      iota = s32[64] iota(), iota_dimension=0
      ROOT bc = s32[8,8] bitcast(iota)
    }

    ENTRY main {
      input = s32[8,8] fusion(), kind=kLoop, calls=input
      init_acc = s32[1,4] constant({{3,2,1,0}})
      c0 = s32[] constant(0)
      tuple = (s32[], s32[8,8], s32[1,4]) tuple(c0, input, init_acc)
      ROOT while = (s32[], s32[8,8], s32[1,4]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"6"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

TEST_F(GpuCopyTest, UseMemcpyForDynamicUpdateSliceWithBitcasts) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> hlo_module,
      ParseAndReturnVerifiedModule(kDynamicUpdateSliceWithBitcastModule));

  CompileAndVerifyIr(std::move(hlo_module), R"(
    CHECK-NOT: void @
    CHECK: void @input
    CHECK-NOT: void @
    CHECK: void @cmp
    CHECK-NOT: void @
    CHECK: void @next_ivar
    CHECK-NOT: void @
  )",
                     /*match_optimized_ir=*/false,
                     /*run_optimization_passes=*/false);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kDynamicUpdateSliceWithBitcastModule,
                                       ErrorSpec{0, 0}));
}

}  // namespace gpu
}  // namespace xla
