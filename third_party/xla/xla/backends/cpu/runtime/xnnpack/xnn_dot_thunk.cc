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
#include <functional>
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
#include "xla/backends/cpu/runtime/xnnpack/parallel_loop_runner.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

// XNNPACK runtime instantiated for the dot operation.
struct XnnDotThunk::XnnRuntime {
  XnnRuntime() = default;
  ~XnnRuntime() { Destroy(); }

  XnnRuntime(XnnRuntime&&);
  XnnRuntime& operator=(XnnRuntime&&);

  absl::Status Invoke(se::DeviceMemoryBase lhs, se::DeviceMemoryBase rhs,
                      se::DeviceMemoryBase out);

  void Destroy();

  xnn_subgraph_t subgraph = nullptr;
  xnn_workspace_t workspace = nullptr;
  xnn_runtime_t runtime = nullptr;

  std::unique_ptr<ParallelLoopRunner> runner;
};

XnnDotThunk::XnnRuntime::XnnRuntime(XnnRuntime&& other) {
  *this = std::move(other);
}

auto XnnDotThunk::XnnRuntime::operator=(XnnRuntime&& other) -> XnnRuntime& {
  Destroy();

  subgraph = other.subgraph;
  workspace = other.workspace;
  runtime = other.runtime;

  other.subgraph = nullptr;
  other.workspace = nullptr;
  other.runtime = nullptr;

  runner = std::move(other.runner);
  return *this;
}

absl::Status XnnDotThunk::XnnRuntime::Invoke(se::DeviceMemoryBase lhs,
                                             se::DeviceMemoryBase rhs,
                                             se::DeviceMemoryBase out) {
  std::array<xnn_external_value, 3> external_values = {
      xnn_external_value{0, lhs.opaque()},
      xnn_external_value{1, rhs.opaque()},
      xnn_external_value{2, out.opaque()},
  };

  XNN_RETURN_IF_ERROR(xnn_setup_runtime_v2(runtime, 3, external_values.data()));
  XNN_RETURN_IF_ERROR(xnn_invoke_runtime(runtime));
  return absl::OkStatus();
}

absl::StatusOr<XnnDotThunk::XnnRuntime> XnnDotThunk::CreateXnnRuntime() {
  VLOG(3) << "Create XNN runtime for dot operation: num_created="
          << xnn_runtime_pool_.num_created();

  XnnRuntime runtime;

  XNN_RETURN_IF_ERROR(xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0,
                                          &runtime.subgraph));

  uint32_t lhs_id = XNN_INVALID_VALUE_ID;
  uint32_t rhs_id = XNN_INVALID_VALUE_ID;
  uint32_t out_id = XNN_INVALID_VALUE_ID;

  auto dims = [](absl::Span<const int64_t> dims) -> std::vector<size_t> {
    return {dims.begin(), dims.end()};
  };

  std::vector<size_t> lhs_dims = dims(dot_slices_.lhs_shape.dimensions());
  std::vector<size_t> rhs_dims = dims(dot_slices_.rhs_shape.dimensions());
  std::vector<size_t> out_dims = dims(dot_slices_.out_shape.dimensions());

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      runtime.subgraph, xnn_datatype_fp32, lhs_dims.size(), lhs_dims.data(),
      nullptr,
      /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &lhs_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      runtime.subgraph, xnn_datatype_fp32, rhs_dims.size(), rhs_dims.data(),
      nullptr,
      /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &rhs_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      runtime.subgraph, xnn_datatype_fp32, out_dims.size(), out_dims.data(),
      nullptr,
      /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id));

  XNN_RETURN_IF_ERROR(xnn_define_batch_matrix_multiply(
      runtime.subgraph, lhs_id, rhs_id, out_id,
      /*flags=*/dot_canonical_dims_.rhs_canonical ? 0 : XNN_FLAG_TRANSPOSE_B));

  XNN_RETURN_IF_ERROR(xnn_create_workspace(&runtime.workspace));

  XNN_RETURN_IF_ERROR(xnn_create_runtime_v4(runtime.subgraph, nullptr,
                                            runtime.workspace, nullptr, 0,
                                            &runtime.runtime));

  XNN_RETURN_IF_ERROR(xnn_reshape_runtime(runtime.runtime));

  return {std::move(runtime)};
}

void XnnDotThunk::XnnRuntime::Destroy() {
  if (runtime != nullptr) XNN_LOG_IF_ERROR(xnn_delete_runtime(runtime));
  if (subgraph != nullptr) XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
  if (workspace != nullptr) XNN_LOG_IF_ERROR(xnn_release_workspace(workspace));
}

absl::StatusOr<bool> XnnDotThunk::IsSupported(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape) {
  // TODO(ezhulenev): Support other element types.
  if (lhs_shape.element_type() != F32 || rhs_shape.element_type() != F32 ||
      out_shape.element_type() != F32) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  // XNNPACK does not support transposing LHS or col-major layouts.
  return dot_canonical_dims.lhs_canonical &&
         !dot_canonical_dims.lhs_column_major &&
         !dot_canonical_dims.rhs_column_major;
}

absl::StatusOr<std::unique_ptr<XnnDotThunk>> XnnDotThunk::Create(
    Info info, DotDimensionNumbers dot_dimensions,
    BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
    BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
    BufferAllocation::Slice out_buffer, Shape out_shape) {
  TF_RETURN_IF_ERROR(InitializeXnnPack());

  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  DotSlices dot_slices{lhs_buffer, std::move(lhs_shape),
                       rhs_buffer, std::move(rhs_shape),
                       out_buffer, std::move(out_shape)};

  return absl::WrapUnique(
      new XnnDotThunk(info, std::move(dot_dimensions), std::move(dot_slices),
                      std::move(dot_shape), std::move(dot_canonical_dims)));
}

XnnDotThunk::XnnDotThunk(Info info, DotDimensionNumbers dot_dimensions,
                         DotSlices dot_slices, DotShape dot_shape,
                         DotCanonicalDims dot_canonical_dims)
    : Thunk(Kind::kXnnDot, info),
      dot_dimensions_(std::move(dot_dimensions)),
      dot_slices_(std::move(dot_slices)),
      dot_shape_(std::move(dot_shape)),
      dot_canonical_dims_(std::move(dot_canonical_dims)),
      xnn_runtime_pool_(std::bind(&XnnDotThunk::CreateXnnRuntime, this)) {}

XnnDotThunk::~XnnDotThunk() = default;

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

  VLOG(3) << absl::StreamFormat(
      "  matmul dims: m=%d, k=%d, n=%d, lhs_column_major=%v, lhs_canonical=%v, "
      "rhs_column_major=%v, rhs_canonical=%v",
      dot_canonical_dims_.m, dot_canonical_dims_.k, dot_canonical_dims_.n,
      dot_canonical_dims_.lhs_column_major, dot_canonical_dims_.lhs_canonical,
      dot_canonical_dims_.rhs_column_major, dot_canonical_dims_.rhs_canonical);

  TF_ASSIGN_OR_RETURN(auto runtime, xnn_runtime_pool_.GetOrCreate());
  TF_RETURN_IF_ERROR(runtime->Invoke(lhs_data, rhs_data, out_data));

  return OkExecuteEvent();
}

}  // namespace xla::cpu
