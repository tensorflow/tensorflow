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

#include "xla/backends/cpu/runtime/xnnpack/xnn_fusion_thunk.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "xnnpack.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
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

static absl::StatusOr<xnn_subgraph_t> CreateBinaryAdd(
    absl::Span<const XnnFusionThunk::Argument> arguments,
    absl::Span<const XnnFusionThunk::Result> results) {
  xnn_subgraph_t subgraph = nullptr;
  XNN_RETURN_IF_ERROR(xnn_create_subgraph(/*external_value_ids=*/3,
                                          /*flags=*/0, &subgraph));

  auto dims = [](absl::Span<const int64_t> dims) -> std::vector<size_t> {
    return {dims.begin(), dims.end()};
  };

  uint32_t lhs_id = XNN_INVALID_VALUE_ID;
  uint32_t rhs_id = XNN_INVALID_VALUE_ID;
  uint32_t out_id = XNN_INVALID_VALUE_ID;

  std::vector<size_t> lhs_dims = dims(arguments[0].shape.dimensions());
  std::vector<size_t> rhs_dims = dims(arguments[1].shape.dimensions());
  std::vector<size_t> out_dims = dims(results[0].shape.dimensions());

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, lhs_dims.size(), lhs_dims.data(), nullptr,
      /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &lhs_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, rhs_dims.size(), rhs_dims.data(), nullptr,
      /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &rhs_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, out_dims.size(), out_dims.data(), nullptr,
      /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id));

  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};

  XNN_RETURN_IF_ERROR(xnn_define_binary(subgraph, xnn_binary_add, &params,
                                        lhs_id, rhs_id, out_id, /*flags=*/0));

  return subgraph;
}

class XnnFusionThunkTest : public testing::TestWithParam<bool> {
 protected:
  bool use_threadpool() const { return GetParam(); }
};

TEST_P(XnnFusionThunkTest, ElementwiseAdd) {
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

  XnnFusionThunk::Argument lhs_arg = {lhs_slice, shape};
  XnnFusionThunk::Argument rhs_arg = {rhs_slice, shape};
  XnnFusionThunk::Result out_res = {out_slice, shape};

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, XnnFusionThunk::Create(
                      XnnFusionThunk::Options{use_threadpool()}, {"fusion"},
                      {lhs_arg, rhs_arg}, {out_res}, &CreateBinaryAdd));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = use_threadpool() ? &device : nullptr;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  EXPECT_EQ(out, LiteralUtil::CreateR1<float>({5.0, 5.0, 5.0, 5.0}));
}

INSTANTIATE_TEST_SUITE_P(XnnFusion, XnnFusionThunkTest,
                         testing::Values(true, false));

}  // namespace
}  // namespace xla::cpu
