/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/ynnpack/ynn_fusion_thunk.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_threadpool.h"
#include "xla/hlo/ir/hlo_instruction.h"
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

static absl::StatusOr<YnnSubgraph> BuildBinaryAddSubgraph(
    absl::Span<const YnnFusionThunk::Argument> arguments,
    absl::Span<const YnnFusionThunk::Result> results) {
  TF_ASSIGN_OR_RETURN(YnnSubgraph subgraph,
                      CreateYnnSubgraph([&](ynn_subgraph_t* subgraph) {
                        return ynn_create_subgraph(
                            /*external_value_ids=*/3,
                            /*flags=*/0, subgraph);
                      }));

  auto dims = [](absl::Span<const int64_t> dims) -> std::vector<size_t> {
    return {dims.begin(), dims.end()};
  };

  uint32_t lhs_id = 0;
  uint32_t rhs_id = 1;
  uint32_t out_id = 2;

  std::vector<size_t> lhs_dims = dims(arguments[0].shape.dimensions());
  std::vector<size_t> rhs_dims = dims(arguments[1].shape.dimensions());
  std::vector<size_t> out_dims = dims(results[0].shape.dimensions());

  YNN_RETURN_IF_ERROR(
      ynn_define_tensor_value(subgraph.get(), ynn_type_fp32, lhs_dims.size(),
                              lhs_dims.data(), /*data=*/nullptr,
                              /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                              /*scale_id=*/YNN_INVALID_VALUE_ID,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &lhs_id));

  YNN_RETURN_IF_ERROR(
      ynn_define_tensor_value(subgraph.get(), ynn_type_fp32, rhs_dims.size(),
                              rhs_dims.data(), /*data=*/nullptr,
                              /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                              /*scale_id=*/YNN_INVALID_VALUE_ID,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &rhs_id));

  YNN_RETURN_IF_ERROR(
      ynn_define_tensor_value(subgraph.get(), ynn_type_fp32, rhs_dims.size(),
                              rhs_dims.data(), /*data=*/nullptr,
                              /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                              /*scale_id=*/YNN_INVALID_VALUE_ID,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id));

  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph.get(), ynn_binary_add, lhs_id,
                                        rhs_id, &out_id, /*flags=*/0));

  return subgraph;
}

class YnnFusionThunkTest : public testing::TestWithParam<bool> {
 public:
  static std::string Name(const ::testing::TestParamInfo<bool>& info) {
    return absl::StrCat(info.param ? "threadpool" : "single_threaded");
  }

 protected:
  bool use_threadpool() const { return GetParam(); }
};

TEST_P(YnnFusionThunkTest, ElementwiseAdd) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  auto lhs = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  auto rhs = LiteralUtil::CreateR1<float>({4.0, 3.0, 2.0, 1.0});
  auto out = LiteralUtil::CreateR1<float>({0.0, 0.0, 0.0, 0.0});

  BufferAllocations allocations = CreateBufferAllocations(lhs, rhs, out);

  auto [lhs_alloc, rhs_alloc, out_alloc] =
      CreateBufferAllocation(lhs, rhs, out);
  auto [lhs_slice, rhs_slice, out_slice] =
      CreateBufferAllocationSlice(lhs_alloc, rhs_alloc, out_alloc);

  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});

  YnnFusionThunk::Argument lhs_arg = {lhs_slice, shape};
  YnnFusionThunk::Argument rhs_arg = {rhs_slice, shape};
  YnnFusionThunk::Result out_res = {out_slice, shape};

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, YnnFusionThunk::Create(
                      YnnFusionThunk::Options{use_threadpool()}, {"fusion"},
                      reinterpret_cast<HloInstruction*>(0xDEADBEEF),
                      {lhs_arg, rhs_arg}, {out_res}, &BuildBinaryAddSubgraph));

  YnnThreadpool threadpool;
  if (use_threadpool()) {
    TF_ASSERT_OK_AND_ASSIGN(threadpool, CreateYnnThreadpool(&device));
  }
  Thunk::YnnParams ynn_params(std::move(threadpool));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = use_threadpool() ? &device : nullptr;
  params.ynn_params = &ynn_params;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  EXPECT_EQ(out, LiteralUtil::CreateR1<float>({5.0, 5.0, 5.0, 5.0}));
}

INSTANTIATE_TEST_SUITE_P(YnnFusion, YnnFusionThunkTest, ::testing::Bool(),
                         YnnFusionThunkTest::Name);

}  // namespace
}  // namespace xla::cpu
