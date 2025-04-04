/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/layout_assignment.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/computation_layout.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;
using ::testing::NotNull;
using ::tsl::testing::IsOkAndHolds;

class LayoutAssignmentTest : public HloTestBase {
 public:
  const se::DeviceDescription& GetDeviceDescription() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }

  se::CudaComputeCapability GetCudaComputeCapability() {
    return GetDeviceDescription().cuda_compute_capability();
  }

  const se::GpuComputeCapability& GetGpuComputeCapability() {
    return GetDeviceDescription().gpu_compute_capability();
  }

  se::dnn::VersionInfo GetDnnVersion() {
    return GetDnnVersionInfoOrDefault(backend().default_stream_executor(),
                                      se::dnn::VersionInfo{8, 9, 0});
  }
};

TEST_F(LayoutAssignmentTest, Elementwise) {
  Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
  Shape ashape_in_row_major(ashape);
  Shape ashape_in_col_major(ashape);
  *ashape_in_row_major.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  *ashape_in_col_major.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  // Enumerate all possible combinations of layouts.
  for (const Shape& lhs_shape_with_layout :
       {ashape_in_row_major, ashape_in_col_major}) {
    for (const Shape& rhs_shape_with_layout :
         {ashape_in_row_major, ashape_in_col_major}) {
      for (const Shape& result_shape_with_layout :
           {ashape_in_row_major, ashape_in_col_major}) {
        // GpuLayoutAssignment should assign the same layout to "add" and its
        // two operands.
        auto builder = HloComputation::Builder(TestName());
        auto x = builder.AddInstruction(
            HloInstruction::CreateParameter(0, ashape, "x"));
        auto y = builder.AddInstruction(
            HloInstruction::CreateParameter(1, ashape, "y"));
        auto add = builder.AddInstruction(
            HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, x, y));
        auto module = CreateNewVerifiedModule();
        HloComputation* computation =
            module->AddEntryComputation(builder.Build(add));

        ComputationLayout computation_layout(
            computation->ComputeProgramShape());
        *computation_layout.mutable_parameter_layout(0) =
            ShapeLayout(lhs_shape_with_layout);
        *computation_layout.mutable_parameter_layout(1) =
            ShapeLayout(rhs_shape_with_layout);
        *computation_layout.mutable_result_layout() =
            ShapeLayout(result_shape_with_layout);

        GpuLayoutAssignment layout_assignment(
            &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
            GetDeviceDescription());
        EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));

        for (const HloInstruction* operand : add->operands()) {
          EXPECT_TRUE(LayoutUtil::Equal(add->shape().layout(),
                                        operand->shape().layout()));
        }
      }
    }
  }
}

TEST_F(LayoutAssignmentTest, DotLayoutUnchangedIfValid) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,2,3]{1,2,0} parameter(0)
    p1 = f32[5,3,4]{1,2,0} parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4]{2,1,0} dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());
  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(F32, {5, 2, 3}, {1, 2, 0}),
                                m::Op().WithShape(F32, {5, 3, 4}, {1, 2, 0}))
                             .WithShape(F32, {5, 2, 4}, {2, 1, 0})));
}

TEST_F(LayoutAssignmentTest, DotLayoutSetToDefaultIfDefaultValid) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,3,2] parameter(0)
    p1 = f32[5,4,3]{0,1,2} parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4] dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={1},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(F32, {5, 3, 2}, {2, 1, 0}),
                                m::Op().WithShape(F32, {5, 4, 3}, {2, 1, 0}))
                             .WithShape(F32, {5, 2, 4}, {2, 1, 0})));
}

TEST_F(LayoutAssignmentTest, DotOperandLayoutSetToBatchRowsColsOtherwise) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[2,3,5]{2,1,0} parameter(0)
    p1 = f32[3,4,5] parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4] dot(p0, p1),
      lhs_batch_dims={2}, lhs_contracting_dims={1},
      rhs_batch_dims={2}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(F32, {2, 3, 5}, {0, 1, 2}),
                                m::Op().WithShape(F32, {3, 4, 5}, {1, 0, 2}))));
}

TEST_F(LayoutAssignmentTest, DotOperandInconsistentDimLayouts) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,6,2,3] parameter(0)
    p1 = f32[6,5,3,4] parameter(1)
    ROOT dot.1330.10585 = f32[5,6,2,4] dot(p0, p1),
      lhs_batch_dims={0,1}, lhs_contracting_dims={3},
      rhs_batch_dims={1,0}, rhs_contracting_dims={2}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(m::Op().WithShape(F32, {5, 6, 2, 3}, {3, 2, 1, 0}),
                        m::Op().WithShape(F32, {6, 5, 3, 4}, {3, 2, 0, 1}))));
}

TEST_F(LayoutAssignmentTest, TransposedDotLayout) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,2,3] parameter(0)
    p1 = f32[5,3,4,6] parameter(1)
    dot = f32[5,2,4,6] dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
    ROOT out = f32[2,5,4,6] transpose(dot), dimensions={1,0,2,3}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Transpose(
                     m::Dot(m::Op().WithShape(F32, {5, 2, 3}, {2, 1, 0}),
                            m::Op().WithShape(F32, {5, 3, 4, 6}, {3, 2, 1, 0}))
                         .WithShape(F32, {5, 2, 4, 6}, {3, 2, 0, 1}))
                     .WithShape(F32, {2, 5, 4, 6}, {3, 2, 1, 0})));
}

TEST_F(LayoutAssignmentTest, TransposedDotOfDotLayout) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[8,50] parameter(0)
    p1 = f32[2,8,4,4] parameter(1)
    p2 = f32[4,38] parameter(2)
    dot.1 = f32[50,2,4,4]{3,2,1,0} dot(p0, p1),
      lhs_contracting_dims={0}, rhs_contracting_dims={1}
    dot.2 = f32[50,2,4,38]{3,2,1,0} dot(dot.1, p2),
      lhs_contracting_dims={2}, rhs_contracting_dims={0}
    ROOT out = f32[2,50,38,4]{2,3,0,1} transpose(dot.2), dimensions={1,0,3,2}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  // The transpose layout is not supported by dot.2. Also, we need a copy
  // between dot.1 and dot.2, because the needed operand layout for the lhs of
  // dot.1 cannot be used as layout for dot.1
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Transpose(
              m::Dot(m::Copy(m::Dot(m::Op().WithShape(F32, {8, 50}, {1, 0}),
                                    m::Op().WithShape(F32, {2, 8, 4, 4},
                                                      {3, 2, 0, 1}))
                                 .WithShape(F32, {50, 2, 4, 4}, {3, 2, 1, 0}))
                         .WithShape(F32, {50, 2, 4, 4}, {3, 1, 0, 2}),
                     m::Op().WithShape(F32, {4, 38}, {1, 0}))
                  .WithShape(F32, {50, 2, 4, 38}, {3, 2, 1, 0}))
              .WithShape(F32, {2, 50, 38, 4}, {2, 3, 0, 1})));
}

TEST_F(LayoutAssignmentTest, DotLayoutS8) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY int8_t {
    p0 = s8[32,64] parameter(0)
    p1 = s8[64,96] parameter(1)
    ROOT out = s32[32,96] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(S8, {32, 64}, {1, 0}),
                                m::Op().WithShape(S8, {64, 96}, {0, 1}))));
}

TEST_F(LayoutAssignmentTest, SameLayoutOnOperandsAndOutputsOfSort) {
  const char* hlo_text = R"(
  HloModule SortLayout

  compare {
    p.0.lhs = f32[] parameter(0)
    p.0.rhs = f32[] parameter(1)
    p.1.lhs = f32[] parameter(2)
    p.1.rhs = f32[] parameter(3)
    ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  ENTRY sort {
    keys = f32[3,2]{0,1} constant({{0,1},{0,1},{0,1}})
    values = f32[2,3]{1,0} parameter(0)
    transpose = f32[3,2]{1,0} transpose(values), dimensions={1,0}
    ROOT sort = (f32[3,2]{1,0}, f32[3,2]{1,0}) sort(keys, transpose),
      dimensions={1}, to_apply=compare
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Sort(m::Op().WithShape(F32, {3, 2}, {1, 0}),
                                 m::Op().WithShape(F32, {3, 2}, {1, 0}))));
}

TEST_F(LayoutAssignmentTest,
       SameLayoutOnOperandsAndOutputsOfCubDeviceRadixSort) {
  const char* hlo_text = R"(
  HloModule SortLayout

  ENTRY sort {
    keys = f32[3,2]{0,1} constant({{0,1},{0,1},{0,1}})
    values = f32[2,3]{1,0} parameter(0)
    transpose = f32[3,2]{1,0} transpose(values), dimensions={1,0}
    ROOT sort = (f32[3,2]{1,0}, f32[3,2]{1,0}, u8[128]{0})
        custom-call(keys, transpose), custom_call_target="__cub$DeviceRadixSort"
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall(m::Op().WithShape(F32, {3, 2}, {1, 0}),
                                       m::Op().WithShape(F32, {3, 2}, {1, 0}))))
      << module->ToString();
}

TEST_F(LayoutAssignmentTest, TopKLayout) {
  const char* hlo_text = R"(
  HloModule topk

  compare-greater-than {
    p.1.lhs.3 = s32[] parameter(2)
    p.1.rhs.4 = s32[] parameter(3)
    p.0.lhs.1 = f32[] parameter(0)
    bitcast-convert = s32[] bitcast-convert(p.0.lhs.1)
    constant = s32[] constant(0)
    compare = pred[] compare(bitcast-convert, constant), direction=LT
    constant.2 = s32[] constant(2147483647)
    xor = s32[] xor(constant.2, bitcast-convert)
    select = s32[] select(compare, xor, bitcast-convert)
    p.0.rhs.2 = f32[] parameter(1)
    bitcast-convert.1 = s32[] bitcast-convert(p.0.rhs.2)
    compare.1 = pred[] compare(bitcast-convert.1, constant), direction=LT
    xor.1 = s32[] xor(constant.2, bitcast-convert.1)
    select.1 = s32[] select(compare.1, xor.1, bitcast-convert.1)
    ROOT compare.5 = pred[] compare(select, select.1), direction=GT
  }

  ENTRY main {
    Arg_0.1 = f32[2048,6]{1,0} parameter(0)
    t = f32[6,2048]{0,1} transpose(Arg_0.1), dimensions={1,0}
    ROOT custom-call.1 = (f32[6,8]{1,0}, s32[6,8]{1,0}) custom-call(t), custom_call_target="__gpu$TopK", api_version=API_VERSION_TYPED_FFI, called_computations={compare-greater-than}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall(
                  m::Transpose(m::Copy().WithShape(F32, {2048, 6}, {0, 1}))
                      .WithShape(F32, {6, 2048}, {1, 0}))));
}

TEST_F(LayoutAssignmentTest, FftLayout) {
  const char* hlo_text = R"(
  HloModule Fft_module

  ENTRY Fft {
    input = c64[8,32]{0,1} parameter(0)
    fft = c64[8,32] fft(input), fft_type=FFT, fft_length={32}
    ROOT transpose = c64[32,8] transpose(fft), dimensions={1,0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(
                  m::Transpose(m::Fft(m::Op().WithShape(C64, {8, 32}, {1, 0}))
                                   .WithShape(C64, {8, 32}, {1, 0})))));
}

TEST_F(LayoutAssignmentTest, CustomCallConstrainedAlias) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = f32[2,5,5]{2,1,0} parameter(0)
  Arg_1 = f32[2,5,5]{2,1,0} parameter(1)
  Arg_2 = f32[2,5,5]{2,1,0} parameter(2)
  dot.0 = f32[2,5,5]{2,1,0} dot(Arg_1, Arg_2), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={2}, operand_precision={highest,highest}
  custom-call.0 = (f32[2,5,5]{1,2,0}, s8[16]{0}, s8[16]{0}) custom-call(Arg_0, dot.0), custom_call_target="dummy_call", operand_layout_constraints={f32[2,5,5]{1,2,0}, f32[2,5,5]{1,2,0}}, output_to_operand_aliasing={{0}: (1, {})}
  ROOT get-tuple-element.0 = f32[2,5,5]{1,2,0} get-tuple-element(custom-call.0), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  auto expect_layout = [](const Shape& shape,
                          absl::Span<const int64_t> minor_to_major) {
    const Layout expected = LayoutUtil::MakeLayout(minor_to_major);
    EXPECT_TRUE(LayoutUtil::Equal(shape.layout(), expected))
        << "Expected layout " << expected << ", actual " << shape.layout();
  };
  expect_layout(ShapeUtil::GetSubshape(call_0->shape(), {0}), {1, 2, 0});
  expect_layout(call_0->operand(0)->shape(), {1, 2, 0});
  expect_layout(call_0->operand(1)->shape(), {1, 2, 0});
}

TEST_F(LayoutAssignmentTest, MoveToHostCustomCallConstrained) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = f32[2,5,5]{0,1,2} parameter(0)
  custom-call.0 = f32[2,5,5] custom-call(Arg_0), custom_call_target="MoveToHost"
  ROOT custom-call.1 = f32[2,5,5]{2, 1, 0} custom-call(custom-call.0),
      custom_call_target="fixed_call", operand_layout_constraints={f32[2,5,5]{1,2,0}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  const Layout input_layout = call_0->operand(0)->shape().layout();
  const Layout output_layout = call_0->shape().layout();
  EXPECT_EQ(input_layout, LayoutUtil::GetDefaultLayoutForR3());
  EXPECT_EQ(output_layout, LayoutUtil::GetDefaultLayoutForR3());
}

TEST_F(LayoutAssignmentTest, MoveToDeviceCustomCallConstrained) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = f32[2,5,5]{1,2,0} parameter(0)
  custom-call.0 = f32[2,5,5] custom-call(Arg_0), custom_call_target="MoveToDevice"
  ROOT custom-call.1 = f32[2,5,5]{2, 1, 0} custom-call(custom-call.0),
      custom_call_target="fixed_call", operand_layout_constraints={f32[2,5,5]{0,1,2}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  const Layout input_layout = call_0->operand(0)->shape().layout();
  const Layout output_layout = call_0->shape().layout();
  EXPECT_EQ(input_layout, LayoutUtil::GetDefaultLayoutForR3());
  EXPECT_EQ(output_layout, LayoutUtil::GetDefaultLayoutForR3());
}

TEST_F(LayoutAssignmentTest, CuDNNConvolutionHasNHWCLayoutPostHopper) {
  const char* hlo = R"(
ENTRY entry {
  p0 = f32[1,64,64,16]{3,2,1,0} parameter(0)
  p1 = f32[3,16,3,32]{3,2,1,0} parameter(1)
  ROOT conv = (f32[1,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call(p0, p1),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b10f_o10i->b10f,
    custom_call_target="__cudnn$convForwardGraph"
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      hlo_module->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(
      &computation_layout, se::CudaComputeCapability::Hopper(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(hlo_module.get()), IsOkAndHolds(true));

  // We start from b10f_o10i->b10f, meaning that the inputs start out as
  // NWHC_OWHI->NWHC. Layout assignment should yield layouts of the form
  // {3,1,2,0} (transpose the middle dimensions) for both inputs and for the
  // output, therefore, in order to get to the desired NHWC_OHWI->NHWC layout.
  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(HloPrintOptions::ShortParsable()), R"(
// CHECK-DAG: [[P0:[^ ]+]] = {{.*}} parameter(0)
// CHECK-DAG: [[P1:[^ ]+]] = {{.*}} parameter(1)
// CHECK-DAG: [[COPY_P0:[^ ]+]] = {{.*}}{3,1,2,0} copy([[P0]])
// CHECK-DAG: [[COPY_P1:[^ ]+]] = {{.*}}{3,1,2,0} copy([[P1]])
// CHECK:     [[CONV:[^ ]+]] = {{.*}}{3,1,2,0}, {{.*}} custom-call([[COPY_P0]], [[COPY_P1]])
)"),
      IsOkAndHolds(true));
}

TEST_F(LayoutAssignmentTest, F64CuDNNConvolutionHasNCHWLayoutPostHopper) {
  const char* hlo = R"(
ENTRY entry {
  p0 = f64[2,64,64,16]{3,2,1,0} parameter(0)
  p1 = f64[6,16,3,32]{3,2,1,0} parameter(1)
  ROOT conv = (f64[2,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call(p0, p1),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b10f_o10i->b10f,
    custom_call_target="__cudnn$convForwardGraph"
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      hlo_module->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(
      &computation_layout, se::CudaComputeCapability::Hopper(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(hlo_module.get()), IsOkAndHolds(true));

  // We start from b10f_o10i->b10f, meaning that the inputs start out as
  // NWHC_OWHI->NWHC. Layout assignment should yield layouts of the form
  // {1,2,3,0} for both inputs and for the output, therefore, in order to get to
  // the desired NCHW_OIHW->NCHW layout.
  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(HloPrintOptions::ShortParsable()), R"(
// CHECK-DAG: [[P0:[^ ]+]] = {{.*}} parameter(0)
// CHECK-DAG: [[P1:[^ ]+]] = {{.*}} parameter(1)
// CHECK-DAG: [[COPY_P0:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P0]])
// CHECK-DAG: [[COPY_P1:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P1]])
// CHECK:     [[CONV:[^ ]+]] = {{.*}}{1,2,3,0}, {{.*}} custom-call([[COPY_P0]], [[COPY_P1]])
)"),
      IsOkAndHolds(true));
}

TEST_F(LayoutAssignmentTest, ConvCuDNNF8) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::kHopper)) {
    GTEST_SKIP() << "FP8 convolutions require HOPPER or newer archiecture.";
  }

  const char* hlo = R"(

  HloModule jit_conv_general_dilated

  ENTRY main.4 {
    Arg_0 = f8e4m3fn[1,64,64,16]{3,2,1,0} parameter(0)
    Arg_1 = f8e4m3fn[3,3,16,32]{3,2,1,0} parameter(1)
    ROOT conv = f8e4m3fn[1,64,64,32]{3,2,1,0} convolution(Arg_0, Arg_1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  }
)";

  MatchOptimizedHlo(hlo, R"(
  // CHECK: [[P0:%[^ ]+]] = f8e4m3fn[1,64,64,16]{3,2,1,0} parameter(0)
  // CHECK: [[P1:%[^ ]+]] = f8e4m3fn[3,3,16,32]{3,2,1,0} parameter(1)
  // CHECK-NEXT: [[P2:%[^ ]+]] = f8e4m3fn[32,3,3,16]{3,2,1,0} transpose([[P1]]), dimensions={3,0,1,2}
  // CHECK-NEXT: [[CONV:%[^ ]+]] = (f8e4m3fn[1,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call([[P0]], [[P2]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
  )");
}

TEST_F(LayoutAssignmentTest, ConvCuDNNBF16) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::kAmpere)) {
    GTEST_SKIP() << "Conv with Bfloat16 uses NHWC layout for "
                    "architectures with Tensor Cores.";
  }

  const char* hlo = R"(

  HloModule jit_conv_general_dilated

  ENTRY main.4 {
    Arg_0.1 = bf16[1,64,64,16]{3,2,1,0} parameter(0), sharding={replicated}
    Arg_1.2 = bf16[3,3,16,32]{3,2,1,0} parameter(1), sharding={replicated}
    ROOT convolution.3 = bf16[1,64,64,32]{3,2,1,0} convolution(Arg_0.1, Arg_1.2), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_name="jit(conv_general_dilated)/jit(main)/conv_general_dilated[window_strides=(1, 1) padding=((1, 1), (1, 1)) lhs_dilation=(1, 1) rhs_dilation=(1, 1) dimension_numbers=ConvDimensionNumbers(lhs_spec=(0, 3, 1, 2), rhs_spec=(3, 2, 0, 1), out_spec=(0, 3, 1, 2)) feature_group_count=1 batch_group_count=1 lhs_shape=(1, 64, 64, 16) rhs_shape=(3, 3, 16, 32) precision=None preferred_element_type=None]" source_file="/usr/local/lib/python3.8/dist-packages/flax/linen/linear.py" source_line=438}
  }
)";

  MatchOptimizedHlo(hlo, R"(
  // CHECK: [[P0:%[^ ]+]] = bf16[1,64,64,16]{3,2,1,0} parameter(0), sharding={replicated}
  // CHECK: [[P1:%[^ ]+]] = bf16[3,3,16,32]{3,2,1,0} parameter(1), sharding={replicated}
  // CHECK-NEXT: [[P2:%[^ ]+]] = bf16[32,3,3,16]{3,2,1,0} transpose([[P1]]), dimensions={3,0,1,2}
  // CHECK-NEXT: %cudnn-conv.1 = (bf16[1,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call([[P0]], [[P2]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward"
  )");
}

TEST_F(LayoutAssignmentTest, ConvCuDNNFP16) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::kVolta)) {
    GTEST_SKIP() << "Conv with FP16 uses NHWC layout for "
                    "architectures with Tensor Cores.";
  }

  const char* hlo = R"(

  HloModule jit_conv_general_dilated

  ENTRY main.4 {
    Arg_0.1 = f16[1,64,64,16]{3,2,1,0} parameter(0), sharding={replicated}
    Arg_1.2 = f16[3,3,16,32]{3,2,1,0} parameter(1), sharding={replicated}
    ROOT convolution.3 = f16[1,64,64,32]{3,2,1,0} convolution(Arg_0.1, Arg_1.2), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  }
)";

  MatchOptimizedHlo(hlo, R"(
  // CHECK: [[P0:%[^ ]+]] = f16[1,64,64,16]{3,2,1,0} parameter(0), sharding={replicated}
  // CHECK: [[P1:%[^ ]+]] = f16[3,3,16,32]{3,2,1,0} parameter(1), sharding={replicated}
  // CHECK-NEXT: [[P2:%[^ ]+]] = f16[32,3,3,16]{3,2,1,0} transpose([[P1]]), dimensions={3,0,1,2}
  // CHECK-NEXT: %cudnn-conv.1 = (f16[1,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call([[P0]], [[P2]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward"
  )");
}

TEST_F(LayoutAssignmentTest, ReduceOperandLayout) {
  const char* module_str = R"(
scalar_add_computation {
  scalar_lhs = c64[] parameter(0)
  scalar_rhs = c64[] parameter(1)
  ROOT add.1 = c64[] add(scalar_lhs, scalar_rhs)
}

ENTRY main {
  param_0 = c64[512,64,1024,32,128]{4,3,2,1,0} parameter(0)
  negate = c64[512,64,1024,32,128]{4,3,2,1,0} negate(param_0)
  constant_7 = c64[] constant((0, 0))
  ROOT reduce.2 = c64[512,1024,128]{2,1,0} reduce(negate, constant_7), dimensions={1,3}, to_apply=scalar_add_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));
  auto reduce = m->entry_computation()->root_instruction();
  EXPECT_EQ(reduce->operand(0)->shape().layout().minor_to_major(),
            LayoutUtil::MakeLayout({3, 1, 4, 2, 0}).minor_to_major());
}

TEST_F(LayoutAssignmentTest, ReduceOperandLayoutDivisorOfWarpSize) {
  // Same as ReduceOperandLayout, but with a small reduction dimension that
  // is a divisor of the warp size.
  const char* module_str = R"(
scalar_add_computation {
  scalar_lhs = c64[] parameter(0)
  scalar_rhs = c64[] parameter(1)
  ROOT add.1 = c64[] add(scalar_lhs, scalar_rhs)
}

ENTRY main {
  param_0 = c64[512,16,1024,128]{3,2,1,0} parameter(0)
  negate = c64[512,16,1024,128]{3,2,1,0} negate(param_0)
  constant_7 = c64[] constant((0, 0))
  ROOT reduce.2 = c64[512,1024,128]{2,1,0} reduce(negate, constant_7), dimensions={1}, to_apply=scalar_add_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));
  auto reduce = m->entry_computation()->root_instruction();
  EXPECT_EQ(reduce->operand(0)->shape().layout().minor_to_major(),
            LayoutUtil::MakeLayout({1, 3, 2, 0}).minor_to_major());
}

TEST_F(LayoutAssignmentTest, AutoLayoutE4M3ContractingMinorFirst) {
  const char* hlo = R"(

  HloModule jit_dot_general_f8e4m3fn

  ENTRY main {
    p0 = f8e4m3fn[128,5120] parameter(0)
    p1 = f8e4m3fn[5120,10240] parameter(1)
    ROOT dot = f32[128,10240] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo, {}, HloParserOptions().set_fill_missing_layouts(false)));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());
  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Dot(m::Parameter(0).WithShape(F8E4M3FN, {128, 5120}, {1, 0}),
                 m::Parameter(1).WithShape(F8E4M3FN, {5120, 10240}, {0, 1}))
              .WithShape(F32, {128, 10240}, {1, 0})));
}

TEST_F(LayoutAssignmentTest, AutoLayoutS4DotContractingMinorLhs) {
  const char* hlo = R"(
  HloModule AutoLayoutS4DotContractingMinorLhs

  ENTRY main {
    p0 = s4[5120,128] parameter(0)
    p0.c = bf16[5120,128] convert(p0)
    p1 = bf16[5120,10240] parameter(1)
    ROOT dot = bf16[128,10240] dot(p0.c, p1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo, {}, HloParserOptions().set_fill_missing_layouts(false)));
  DebugOptions debug_options = m->config().debug_options();
  debug_options.set_xla_gpu_experimental_pack_dot_operands_along_k_dimension(
      true);
  m->mutable_config().set_debug_options(debug_options);
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());
  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));
  EXPECT_THAT(m->entry_computation()->parameter_instruction(0),
              GmockMatch(m::Parameter(0).WithShape(S4, {5120, 128}, {0, 1})));
  EXPECT_THAT(
      m->entry_computation()->parameter_instruction(1),
      GmockMatch(m::Parameter(1).WithShape(BF16, {5120, 10240}, {1, 0})));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot().WithShape(BF16, {128, 10240}, {1, 0})));
}

TEST_F(LayoutAssignmentTest, AutoLayoutS4DotContractingMinorRhs) {
  const char* hlo = R"(
  HloModule AutoLayoutS4DotContractingMinorRhs

  ENTRY main {
    p0 = bf16[5120,128] parameter(0)
    p1 = s4[5120,10240] parameter(1)
    p1.c = bf16[5120,10240] convert(p1)
    ROOT dot = bf16[128,10240] dot(p0, p1.c), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo, {}, HloParserOptions().set_fill_missing_layouts(false)));
  DebugOptions debug_options = m->config().debug_options();
  debug_options.set_xla_gpu_experimental_pack_dot_operands_along_k_dimension(
      true);
  m->mutable_config().set_debug_options(debug_options);
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());
  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));
  EXPECT_THAT(m->entry_computation()->parameter_instruction(0),
              GmockMatch(m::Parameter(0).WithShape(BF16, {5120, 128}, {1, 0})));
  EXPECT_THAT(m->entry_computation()->parameter_instruction(1),
              GmockMatch(m::Parameter(1).WithShape(S4, {5120, 10240}, {0, 1})));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot().WithShape(BF16, {128, 10240}, {1, 0})));
}

TEST_F(LayoutAssignmentTest, VariadicReduceSameOperandLayout) {
  const char* module_str = R"(
HloModule variadic_reduce

reducer {
  %Arg_0.261 = s32[] parameter(0)
  %Arg_2.263 = s32[] parameter(2)
  mul = s32[] multiply(%Arg_0.261, %Arg_2.263)
  %Arg_1.260 = f32[] parameter(1)
  %Arg_3.262 = f32[] parameter(3)
  add = f32[] add(%Arg_1.260, %Arg_3.262)
  ROOT %tuple = (s32[], f32[]) tuple(mul, add)
}

ENTRY main {
  param_0 = f32[512,16,1024,128]{3,2,1,0} parameter(0)
  param_1 = s32[128,1024,16,512]{3,2,1,0} parameter(1)
  transpose = s32[512,16,1024,128]{0,1,2,3} transpose(param_1), dimensions={3,2,1,0}
  zero = f32[] constant(0.0)
  one = s32[] constant(1)
  ROOT reduce.2 = (s32[512,1024,128]{0,1,2}, f32[512,1024,128]{2,1,0}) reduce(transpose, param_0, one, zero), dimensions={1}, to_apply=reducer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));
  auto reduce = m->entry_computation()->root_instruction();
  EXPECT_EQ(reduce->operand(0)->shape().layout().minor_to_major(),
            reduce->operand(1)->shape().layout().minor_to_major());
}

TEST_F(LayoutAssignmentTest, SendRcvLayout) {
  const char* hlo = R"(
HloModule Module

condition  {
    p = (f32[100,100], (f32[100,100], u32[], token[])) parameter(0)
    ROOT lt = pred[] constant(1)
}

body {
    p = (f32[100,100], (f32[100,100], u32[], token[])) parameter(0)

    t1 = f32[100,100] get-tuple-element(p), index=0
    t = (f32[100,100], u32[], token[]) get-tuple-element(p), index=1
    sdone = token[] send-done(t), channel_id=3, frontend_attributes={
      _xla_send_recv_pipeline="0"
    }
    tk = token[] after-all()


    rcvd = (f32[100,100]{0,1}, u32[], token[]) recv(tk), channel_id=2
    zz = (f32[100,100]{0,1}, token[]) recv-done(rcvd), channel_id=2

    rcvd_d = get-tuple-element(zz), index=0

    snd = (f32[100,100]{0,1}, u32[], token[]) send(t1, tk), channel_id=3, frontend_attributes={
      _xla_send_recv_pipeline="0"
    }
    a = add(t1, t1)

    b = add(rcvd_d, a)

    ROOT tup =  tuple(b, snd)
}

ENTRY %main {
    p0 = f32[100,100] parameter(0)
    tk = token[] after-all()
    snd = (f32[100,100]{0,1}, u32[], token[]) send(p0, tk), channel_id=1, frontend_attributes={
      _xla_send_recv_pipeline="0"
    }
    t = tuple(p0, snd)
    ROOT loop = while(t), condition=condition, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  RunAndFilecheckHloRewrite(
      hlo,
      GpuLayoutAssignment{&computation_layout, GetGpuComputeCapability(),
                          GetDnnVersion(), GetDeviceDescription()},
      R"(
// CHECK: (f32[100,100]{1,0}, u32[], token[]) recv
// CHECK:  (f32[100,100]{1,0}, token[]) recv-done
// CHECK:  (f32[100,100]{1,0}, u32[], token[]) send
                                )");
}

TEST_F(LayoutAssignmentTest, RaggedAllToAllLayoutSetRaggedDimToMajor) {
  absl::string_view hlo = R"(
  HloModule module

  ENTRY main {
    input = f32[16,4,8]{0,2,1} parameter(0)
    output = f32[32,4,8]{0,1,2} parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[32,4,8]{0,1,2} ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1}}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(), /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(
      &computation_layout, GetGpuComputeCapability(), GetDnnVersion(),
      GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));
  auto ragged_all_to_all = FindInstruction(m.get(), HloOpcode::kRaggedAllToAll);
  EXPECT_THAT(ragged_all_to_all, NotNull());

  // Operands 0 and 1 of ragged-all-to-all can have different shapes, but they
  // must have the same layout.
  EXPECT_EQ(ragged_all_to_all->operand(0)->shape().layout().minor_to_major(),
            ragged_all_to_all->operand(1)->shape().layout().minor_to_major());

  // Operand 1 is aliased to the output of the ragged-all-to-all, so they must
  // have the same shape.
  EXPECT_EQ(ragged_all_to_all->operand(1)->shape(), ragged_all_to_all->shape());

  // The ragged dimension (0) must be in the most major position in the layout.
  EXPECT_TRUE(ShapeUtil::IsEffectivelyMostMajorDimension(
      ragged_all_to_all->shape(), 0));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
