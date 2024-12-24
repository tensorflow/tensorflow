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
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

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

TEST(XnnFusionThunkTest, ElementwiseAdd) {
  std::vector<MaybeOwningDeviceMemory> buffers;

  std::vector<float> lhs = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> rhs = {4.0, 3.0, 2.0, 1.0};
  std::vector<float> out(4, 0.0);

  size_t size_in_bytes = lhs.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(lhs.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(rhs.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(out.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation lhs_alloc(0, size_in_bytes, 0);
  BufferAllocation rhs_alloc(1, size_in_bytes, 0);
  BufferAllocation out_alloc(2, size_in_bytes, 0);

  BufferAllocation::Slice lhs_slice(&lhs_alloc, 0, size_in_bytes);
  BufferAllocation::Slice rhs_slice(&rhs_alloc, 0, size_in_bytes);
  BufferAllocation::Slice out_slice(&out_alloc, 0, size_in_bytes);

  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});

  XnnFusionThunk::Argument lhs_arg = {lhs_slice, shape};
  XnnFusionThunk::Argument rhs_arg = {rhs_slice, shape};
  XnnFusionThunk::Result out_res = {out_slice, shape};

  TF_ASSERT_OK_AND_ASSIGN(auto thunk,
                          XnnFusionThunk::Create({"fusion"}, {lhs_arg, rhs_arg},
                                                 {out_res}, &CreateBinaryAdd));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  std::vector<float> expected = {5.0, 5.0, 5.0, 5.0};
  EXPECT_EQ(out, expected);
}

}  // namespace
}  // namespace xla::cpu
