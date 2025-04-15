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

#include "xla/backends/cpu/runtime/xnnpack/xnn_dot_thunk.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

struct XnnDotThunkTestSpec {
  PrimitiveType input_type;
  bool use_threadpool;
};

class XnnDotThunkTest : public testing::TestWithParam<XnnDotThunkTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<XnnDotThunkTestSpec>& info) {
    return absl::StrCat(
        PrimitiveType_Name(info.param.input_type), "_",
        info.param.use_threadpool ? "threadpool" : "single_threaded");
  }
};

TEST_P(XnnDotThunkTest, SimpleDot) {
  XnnDotThunkTestSpec spec = GetParam();
  if (spec.input_type == BF16 &&
      !tsl::port::TestCPUFeature(tsl::port::AVX512_BF16)) {
    GTEST_SKIP() << "CPU needs AVX512_BF16 for this test.";
  }
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  auto lhs = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto rhs = LiteralUtil::CreateR2<float>({{4.0, 3.0}, {2.0, 1.0}});
  auto out = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});
  if (spec.input_type == BF16) {
    lhs = LiteralUtil::ConvertF32ToBF16(lhs);
    rhs = LiteralUtil::ConvertF32ToBF16(rhs);
  }

  BufferAllocations allocations = CreateBufferAllocations(lhs, rhs, out);

  auto [lhs_alloc, rhs_alloc, out_alloc] =
      CreateBufferAllocation(lhs, rhs, out);
  auto [lhs_slice, rhs_slice, out_slice] =
      CreateBufferAllocationSlice(lhs_alloc, rhs_alloc, out_alloc);

  Shape input_shape = ShapeUtil::MakeShape(spec.input_type, {2, 2});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 2});

  DotDimensionNumbers dot_dimensions;
  dot_dimensions.add_lhs_contracting_dimensions(1);
  dot_dimensions.add_rhs_contracting_dimensions(0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      XnnDotThunk::Create(XnnDotThunk::Options{spec.use_threadpool}, {"dot"},
                          dot_dimensions, lhs_slice, input_shape, rhs_slice,
                          input_shape, out_slice, output_shape));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = spec.use_threadpool ? &device : nullptr;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  EXPECT_EQ(out, LiteralUtil::CreateR2<float>({{8.0, 5.0}, {20.0, 13.0}}));
}

std::vector<XnnDotThunkTestSpec> GetXnnDotThunkTestSpecs() {
  return std::vector<XnnDotThunkTestSpec>{
      XnnDotThunkTestSpec{F32, /*use_threadpool=*/true},
      XnnDotThunkTestSpec{F32, /*use_threadpool=*/false},
      XnnDotThunkTestSpec{BF16, /*use_threadpool=*/true},
      XnnDotThunkTestSpec{BF16, /*use_threadpool=*/false}};
}

INSTANTIATE_TEST_SUITE_P(XnnDot, XnnDotThunkTest,
                         ::testing::ValuesIn(GetXnnDotThunkTestSpecs()),
                         XnnDotThunkTest::Name);

}  // namespace
}  // namespace xla::cpu
