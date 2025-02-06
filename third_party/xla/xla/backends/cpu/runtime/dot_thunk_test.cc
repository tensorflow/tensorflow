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

#include <cstdint>

#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(DotThunkTest, SimpleDot) {
  auto lhs = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto rhs = LiteralUtil::CreateR2<float>({{4.0, 3.0}, {2.0, 1.0}});
  auto out = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(lhs, rhs, out);

  auto [lhs_alloc, rhs_alloc, out_alloc] =
      CreateBufferAllocation(lhs, rhs, out);
  auto [lhs_slice, rhs_slice, out_slice] =
      CreateBufferAllocationSlice(lhs_alloc, rhs_alloc, out_alloc);

  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});

  DotDimensionNumbers dot_dimensions;
  dot_dimensions.add_lhs_contracting_dimensions(1);
  dot_dimensions.add_rhs_contracting_dimensions(0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, DotThunk::Create({"dot"}, dot_dimensions, lhs_slice, shape,
                                   rhs_slice, shape, out_slice, shape));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  EXPECT_EQ(out, LiteralUtil::CreateR2<float>({{8.0, 5.0}, {20.0, 13.0}}));
}

TEST(DotThunkTest, ThreadedDot) {
  auto shape = ShapeUtil::MakeShape(F32, {1024, 1024});
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

}  // namespace
}  // namespace xla::cpu
