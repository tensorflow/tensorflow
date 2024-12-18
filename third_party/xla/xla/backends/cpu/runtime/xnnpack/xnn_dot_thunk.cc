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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

static absl::Status DefineXnnSubgraph(xnn_subgraph_t subgraph,
                                      const DotDimensionNumbers& dot_dimensions,
                                      const DotShape& dot_shape) {
  uint32_t lhs_id = XNN_INVALID_VALUE_ID;
  uint32_t rhs_id = XNN_INVALID_VALUE_ID;
  uint32_t out_id = XNN_INVALID_VALUE_ID;

  auto dims = [](absl::Span<const int64_t> dims) -> std::vector<size_t> {
    return {dims.begin(), dims.end()};
  };

  std::vector<size_t> lhs_dims = dims(dot_shape.lhs_matmul_shape.dimensions());
  std::vector<size_t> rhs_dims = dims(dot_shape.rhs_matmul_shape.dimensions());
  std::vector<size_t> out_dims = dims(dot_shape.out_matmul_shape.dimensions());

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, lhs_dims.size(), lhs_dims.data(), nullptr,
      /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &lhs_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, rhs_dims.size(), rhs_dims.data(), nullptr,
      /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &rhs_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, out_dims.size(), out_dims.data(), nullptr,
      /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id));

  XNN_RETURN_IF_ERROR(xnn_define_batch_matrix_multiply(subgraph, lhs_id, rhs_id,
                                                       out_id,
                                                       /*flags=*/0));

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<XnnDotThunk>> XnnDotThunk::Create(
    Info info, DotDimensionNumbers dot_dimensions,
    BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
    BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
    BufferAllocation::Slice out_buffer, Shape out_shape) {
  TF_RETURN_IF_ERROR(InitializeXnnPack());

  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  DotSlices dot_slices{lhs_buffer, std::move(lhs_shape),
                       rhs_buffer, std::move(rhs_shape),
                       out_buffer, std::move(out_shape)};

  return absl::WrapUnique(new XnnDotThunk(info, std::move(dot_dimensions),
                                          std::move(dot_slices),
                                          std::move(dot_shape)));
}

XnnDotThunk::XnnDotThunk(Info info, DotDimensionNumbers dot_dimensions,
                         DotSlices dot_slices, DotShape dot_shape)
    : Thunk(Kind::kXnnDot, info),
      dot_dimensions_(std::move(dot_dimensions)),
      dot_slices_(std::move(dot_slices)),
      dot_shape_(std::move(dot_shape)) {}

tsl::AsyncValueRef<XnnDotThunk::ExecuteEvent> XnnDotThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase lhs_data,
      params.buffer_allocations->GetDeviceAddress(dot_slices_.lhs_buffer));

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase rhs_data,
      params.buffer_allocations->GetDeviceAddress(dot_slices_.rhs_buffer));

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase out_data,
      params.buffer_allocations->GetDeviceAddress(dot_slices_.out_buffer));

  VLOG(3) << absl::StreamFormat(
      "XNN dot operation: lhs_batch_dims=[%s], rhs_batch_dims=[%s], "
      "lhs_contract_dims=[%s], rhs_contract_dims=[%s]",
      absl::StrJoin(dot_dimensions_.lhs_batch_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.rhs_batch_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.lhs_contracting_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.rhs_contracting_dimensions(), ","));

  VLOG(3) << absl::StreamFormat(
      "  lhs: %s in slice %s (%p)", dot_slices_.lhs_shape.ToString(true),
      dot_slices_.lhs_buffer.ToString(), lhs_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "  rhs: %s in slice %s (%p)", dot_slices_.rhs_shape.ToString(true),
      dot_slices_.rhs_buffer.ToString(), rhs_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "  out: %s in slice %s (%p)", dot_slices_.out_shape.ToString(true),
      dot_slices_.out_buffer.ToString(), out_data.opaque());

  VLOG(3) << absl::StreamFormat(
      "  matmul shape: batch_size=%d, lhs=%s, rhs=%s, out=%s",
      dot_shape_.batch_size, dot_shape_.lhs_matmul_shape.ToString(true),
      dot_shape_.rhs_matmul_shape.ToString(true),
      dot_shape_.out_matmul_shape.ToString(true));

  xnn_subgraph_t subgraph = nullptr;
  XNN_RETURN_IF_ERROR(
      xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));

  TF_RETURN_IF_ERROR(DefineXnnSubgraph(subgraph, dot_dimensions_, dot_shape_));

  xnn_workspace_t workspace = nullptr;
  XNN_RETURN_IF_ERROR(xnn_create_workspace(&workspace));

  xnn_runtime_t runtime = nullptr;
  XNN_RETURN_IF_ERROR(xnn_create_runtime_v4(subgraph, nullptr, workspace,
                                            nullptr, 0, &runtime));

  std::array<xnn_external_value, 3> external_values = {
      xnn_external_value{0, lhs_data.opaque()},
      xnn_external_value{1, rhs_data.opaque()},
      xnn_external_value{2, out_data.opaque()},
  };

  XNN_RETURN_IF_ERROR(xnn_reshape_runtime(runtime));
  XNN_RETURN_IF_ERROR(xnn_setup_runtime_v2(runtime, 3, external_values.data()));

  XNN_RETURN_IF_ERROR(xnn_invoke_runtime(runtime));

  XNN_RETURN_IF_ERROR(xnn_delete_runtime(runtime));
  XNN_RETURN_IF_ERROR(xnn_delete_subgraph(subgraph));
  XNN_RETURN_IF_ERROR(xnn_release_workspace(workspace));

  return OkExecuteEvent();
}

}  // namespace xla::cpu
