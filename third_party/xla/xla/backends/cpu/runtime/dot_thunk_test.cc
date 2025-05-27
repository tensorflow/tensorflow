/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/dot_thunk.h"

#include <tuple>

#include "absl/strings/str_cat.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

class DotThunkLayoutTest
    : public testing::TestWithParam<std::tuple<bool, bool, bool, bool>> {};

TEST_P(DotThunkLayoutTest, SimpleDot) {
  Layout row_major_layout = LayoutUtil::MakeLayout({1, 0});
  Layout column_major_layout = LayoutUtil::MakeLayout({0, 1});

  const auto& [lhs_col_major, rhs_col_major, out_col_major, canonical] =
      GetParam();

  auto lhs = LiteralUtil::CreateR2WithLayout<float>(
      {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
      lhs_col_major ? column_major_layout : row_major_layout);
  auto rhs = LiteralUtil::CreateR2WithLayout<float>(
      {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}},
      rhs_col_major ? column_major_layout : row_major_layout);
  auto out = canonical
                 ? LiteralUtil::CreateR2WithLayout<float>(
                       {{0.0, 0.0}, {0.0, 0.0}},
                       out_col_major ? column_major_layout : row_major_layout)
                 : LiteralUtil::CreateR2WithLayout<float>(
                       {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
                       out_col_major ? column_major_layout : row_major_layout);

  BufferAllocations allocations = CreateBufferAllocations(lhs, rhs, out);

  auto [lhs_alloc, rhs_alloc, out_alloc] =
      CreateBufferAllocation(lhs, rhs, out);
  auto [lhs_slice, rhs_slice, out_slice] =
      CreateBufferAllocationSlice(lhs_alloc, rhs_alloc, out_alloc);

  DotDimensionNumbers dot_dimensions;
  if (canonical) {
    dot_dimensions.add_lhs_contracting_dimensions(1);
    dot_dimensions.add_rhs_contracting_dimensions(0);
  } else {
    dot_dimensions.add_lhs_contracting_dimensions(0);
    dot_dimensions.add_rhs_contracting_dimensions(1);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      DotThunk::Create({"dot"}, dot_dimensions, lhs_slice, lhs.shape(),
                       rhs_slice, rhs.shape(), out_slice, out.shape()));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  if (canonical) {
    EXPECT_EQ(out,
              LiteralUtil::CreateR2<float>({{58.0, 64.0}, {139.0, 154.0}}));
  } else {
    EXPECT_EQ(out, LiteralUtil::CreateR2<float>({{39.0, 49.0, 59.0},
                                                 {54.0, 68.0, 82.0},
                                                 {69.0, 87.0, 105.0}}));
  }
}

TEST(DotThunkTest, ThreadedDot) {
  auto shape = ShapeUtil::MakeValidatedShape(F32, {1024, 1024}).value();
  // These aren't very interesting literals, but they should be large enough to
  // trigger multi-threaded execution.
  auto lhs = *LiteralUtil::CreateLiteralWithGenerator<F32, float>(
      shape, [](auto) { return 1.0; });
  auto rhs = *LiteralUtil::CreateLiteralWithGenerator<F32, float>(
      shape, [](auto) { return 1.0; });
  auto out = *LiteralUtil::CreateLiteralWithGenerator<F32, float>(
      shape, [](auto) { return 0; });

  BufferAllocations allocations = CreateBufferAllocations(lhs, rhs, out);

  auto [lhs_alloc, rhs_alloc, out_alloc] =
      CreateBufferAllocation(lhs, rhs, out);
  auto [lhs_slice, rhs_slice, out_slice] =
      CreateBufferAllocationSlice(lhs_alloc, rhs_alloc, out_alloc);

  DotDimensionNumbers dot_dimensions;
  dot_dimensions.add_lhs_contracting_dimensions(1);
  dot_dimensions.add_rhs_contracting_dimensions(0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, DotThunk::Create({"dot"}, dot_dimensions, lhs_slice, shape,
                                   rhs_slice, shape, out_slice, shape));

  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  auto expected = *LiteralUtil::CreateLiteralWithGenerator<F32, float>(
      shape, [&](auto) { return shape.dimensions(0); });
  EXPECT_EQ(out, expected);
}

INSTANTIATE_TEST_SUITE_P(
    DotThunkLayoutTest, DotThunkLayoutTest,
    testing::Combine(testing::Bool(), testing::Bool(), testing::Bool(),
                     testing::Bool()),
    [](const testing::TestParamInfo<DotThunkLayoutTest::ParamType>& info) {
      return absl::StrCat(
          std::get<0>(info.param) ? "col_major" : "lhs_row_major", "__",
          std::get<1>(info.param) ? "rhs_col_major" : "rhs_row_major", "__",
          std::get<2>(info.param) ? "out_col_major" : "out_row_major", "__",
          std::get<3>(info.param) ? "canonical" : "non_canonical");
    });

}  // namespace
}  // namespace xla::cpu
