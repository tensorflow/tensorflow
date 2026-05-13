/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/dynamic_slice_fusion_v2_thunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static int64_t ComputeSliceOffset(const DynamicSliceConfig& config,
                                  absl::Span<const WhileLoopState> loop_nest) {
  int64_t iteration = 0;
  if (config.has_loop_index() && config.loop_index() < loop_nest.size()) {
    iteration =
        loop_nest[loop_nest.size() - 1 - config.loop_index()].loop_iteration;
  }
  return config.byte_offset() + iteration * config.byte_stride();
}

// Computes the byte offset from actual offset scalars (D2H-copied from
// device), using the same clamping logic as XLA:
//   start_index = clamp(index, 0, src_dim - dst_dim)
//   byte_offset = sum(start_index * byte_stride)
static int64_t ComputeByteOffsetFromScalars(
    absl::Span<const int32_t> indices, const Shape& src_shape,
    const Shape& dst_shape,
    absl::Span<const DynamicSliceFusion::Offset> offsets) {
  auto byte_strides = ShapeUtil::ByteStrides(src_shape);
  int64_t byte_offset = 0;
  size_t runtime_idx = 0;
  for (const auto& offset : offsets) {
    auto* runtime = std::get_if<DynamicSliceFusion::RuntimeOffset>(&offset);
    if (runtime == nullptr) {
      continue;
    }
    int64_t dim = runtime->dimension_number;
    int64_t idx = indices[runtime_idx++];
    int64_t max_valid = src_shape.dimensions(dim) - dst_shape.dimensions(dim);
    int64_t clamped = std::min(std::max(idx, int64_t{0}), max_valid);
    byte_offset += clamped * (*byte_strides)[dim];
  }
  return byte_offset;
}

//===----------------------------------------------------------------------===//
// DynamicSliceFusionV2Thunk
//===----------------------------------------------------------------------===//

DynamicSliceFusionV2Thunk::DynamicSliceFusionV2Thunk(
    ThunkInfo thunk_info, std::vector<DynamicSliceFusion::Parameter> parameters,
    std::vector<DynamicSliceFusion::Result> results,
    std::vector<BufferAllocation::Slice> parameter_buffers,
    std::vector<BufferAllocation::Slice> result_buffers,
    std::vector<BufferAllocation> slice_allocations,
    ThunkSequence embedded_thunks, bool verify_offsets)
    : Thunk(Kind::kDynamicSliceFusion, std::move(thunk_info)),
      parameters_(std::move(parameters)),
      results_(std::move(results)),
      parameter_buffers_(std::move(parameter_buffers)),
      result_buffers_(std::move(result_buffers)),
      slice_allocations_(std::move(slice_allocations)),
      executor_(std::move(embedded_thunks)),
      verify_offsets_(verify_offsets) {
  DCHECK_EQ(parameter_buffers_.size(), parameters_.size());
  DCHECK_EQ(result_buffers_.size(), results_.size());
}

std::string DynamicSliceFusionV2Thunk::ToString(int indent) const {
  std::string result;
  absl::StrAppendFormat(&result, "params=%d, results=%d\n", parameters_.size(),
                        results_.size());
  absl::StrAppend(&result, executor_.thunks().ToString(indent + 1));
  return result;
}

absl::Status DynamicSliceFusionV2Thunk::Prepare(const PrepareParams& params) {
  return executor_.Prepare(params);
}

absl::Status DynamicSliceFusionV2Thunk::Initialize(
    const InitializeParams& params) {
  return executor_.Initialize(params);
}

static absl::Status VerifySliceOffset(
    se::Stream& stream, const BufferAllocations& orig, absl::string_view kind,
    size_t idx, const std::optional<DynamicSliceConfig>& config,
    const std::optional<std::vector<DynamicSliceFusion::Offset>>& offsets,
    const Shape& src_shape, const Shape& dst_shape,
    absl::Span<const WhileLoopState> loop_nest,
    absl::Span<const DynamicSliceFusion::Parameter> parameters,
    absl::Span<const BufferAllocation::Slice> parameter_buffers) {
  if (!config.has_value() || !offsets.has_value()) {
    return absl::OkStatus();
  }

  // Collect offsets that depend on runtime values.
  std::vector<DynamicSliceFusion::RuntimeOffset> runtime_offsets;
  for (const auto& offset : *offsets) {
    if (auto* rt = std::get_if<DynamicSliceFusion::RuntimeOffset>(&offset)) {
      runtime_offsets.push_back(*rt);
    }
  }

  // Copy offsets to host (offset buffers must be scalars).
  std::vector<int32_t> indices(runtime_offsets.size());
  for (size_t d = 0; d < runtime_offsets.size(); ++d) {
    int64_t param_num = runtime_offsets[d].parameter_number;
    if (!ShapeUtil::IsScalarWithElementType(
            parameters[param_num].parameter_shape, S32)) {
      return Internal(
          "Expected S32 scalar offset parameter at index %d, got %s", param_num,
          ShapeUtil::HumanString(parameters[param_num].parameter_shape));
    }
    auto src = orig.GetDeviceAddress(parameter_buffers[param_num]);
    RETURN_IF_ERROR(stream.Memcpy(&indices[d], src, sizeof(int32_t)));
  }

  // Wait for completion of all memory copies.
  if (!indices.empty()) {
    RETURN_IF_ERROR(stream.BlockHostUntilDone());
  }

  // Check that the value passed at run time matches statically computed offset.
  int64_t actual_offset =
      ComputeByteOffsetFromScalars(indices, src_shape, dst_shape, *offsets);
  int64_t annotated_offset = ComputeSliceOffset(*config, loop_nest);

  if (actual_offset != annotated_offset) {
    return Internal(
        "Dynamic slice fusion offset mismatch for %s[%d]: "
        "annotated offset=%d vs actual offset=%d",
        kind, idx, annotated_offset, actual_offset);
  }
  return absl::OkStatus();
}

static absl::Status VerifyOffsets(
    const Thunk::ExecuteParams& params,
    absl::Span<const WhileLoopState> loop_nest,
    absl::Span<const DynamicSliceFusion::Parameter> parameters,
    absl::Span<const DynamicSliceFusion::Result> results,
    absl::Span<const BufferAllocation::Slice> parameter_buffers) {
  se::Stream& stream = *params.stream;
  const BufferAllocations& orig = *params.buffer_allocations;

  for (size_t i = 0; i < parameters.size(); ++i) {
    RETURN_IF_ERROR(VerifySliceOffset(
        stream, orig, "param", i, parameters[i].slice_config,
        parameters[i].slice_offsets, parameters[i].parameter_shape,
        parameters[i].slice_shape, loop_nest, parameters, parameter_buffers));
  }

  for (size_t j = 0; j < results.size(); ++j) {
    RETURN_IF_ERROR(VerifySliceOffset(
        stream, orig, "result", j, results[j].update_config,
        results[j].update_offsets, results[j].result_shape,
        results[j].update_shape, loop_nest, parameters, parameter_buffers));
  }

  return absl::OkStatus();
}

absl::Status DynamicSliceFusionV2Thunk::ExecuteOnStream(
    const ExecuteParams& params) {
  absl::Span<const WhileLoopState> loop_nest = IsInsideWhileLoopNest();
  const BufferAllocations& orig = *params.buffer_allocations;

  XLA_VLOG_DEVICE(3, orig.device_ordinal()) << absl::StrFormat(
      "Dynamic slice fusion: %d parameters, %d results, loop_nest depth=%d",
      parameters_.size(), results_.size(), loop_nest.size());

  if (verify_offsets_) {
    RETURN_IF_ERROR(VerifyOffsets(params, loop_nest, parameters_, results_,
                                  parameter_buffers_));
  }

  std::vector<se::DeviceAddressBase> buffers =
      BuildDynamicSliceBuffers(orig, loop_nest);

  BufferAllocations slice_allocations(buffers, orig.device_ordinal(),
                                      orig.memory_allocator());
  ExecuteParams dynamic_slice_params =
      ExecuteParams::CloneWithNewAllocations(params, slice_allocations);
  return executor_.ExecuteOnStream(dynamic_slice_params);
}

std::vector<se::DeviceAddressBase>
DynamicSliceFusionV2Thunk::BuildDynamicSliceBuffers(
    const BufferAllocations& orig,
    absl::Span<const WhileLoopState> loop_nest) const {
  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(parameters_.size() + results_.size());

  for (size_t i = 0; i < parameters_.size(); ++i) {
    se::DeviceAddressBase addr = orig.GetDeviceAddress(parameter_buffers_[i]);
    int64_t sliced_size = slice_allocations_[i].size();
    if (parameters_[i].slice_config.has_value()) {
      int64_t offset =
          ComputeSliceOffset(*parameters_[i].slice_config, loop_nest);
      XLA_VLOG_DEVICE(3, orig.device_ordinal()) << absl::StrFormat(
          "  param[%d]: base=%p size=%d -> offset=%d sliced_size=%d", i,
          addr.opaque(), addr.size(), offset, sliced_size);
      addr = addr.GetByteSlice(offset, sliced_size);
    }
    buffers.push_back(addr);
  }

  for (size_t j = 0; j < results_.size(); ++j) {
    se::DeviceAddressBase addr = orig.GetDeviceAddress(result_buffers_[j]);
    int64_t sliced_size = slice_allocations_[parameters_.size() + j].size();
    if (results_[j].update_config.has_value()) {
      int64_t offset =
          ComputeSliceOffset(*results_[j].update_config, loop_nest);
      XLA_VLOG_DEVICE(3, orig.device_ordinal()) << absl::StrFormat(
          "  result[%d]: base=%p size=%d -> offset=%d sliced_size=%d", j,
          addr.opaque(), addr.size(), offset, sliced_size);
      addr = addr.GetByteSlice(offset, sliced_size);
    }
    buffers.push_back(addr);
  }

  return buffers;
}

Thunk::BufferUses DynamicSliceFusionV2Thunk::buffer_uses() const {
  BufferUses uses;
  for (size_t i = 0; i < parameters_.size(); ++i) {
    uses.push_back(
        BufferUse::Read(parameter_buffers_[i], parameters_[i].parameter_shape));
  }
  for (size_t j = 0; j < results_.size(); ++j) {
    uses.push_back(
        BufferUse::Write(result_buffers_[j], results_[j].result_shape));
  }
  return uses;
}

absl::Status DynamicSliceFusionV2Thunk::WalkNested(Walker callback) {
  return executor_.thunks().WalkNested(callback);
}

absl::Status DynamicSliceFusionV2Thunk::TransformNested(Transformer callback) {
  return executor_.thunks().TransformNested(callback);
}

//===----------------------------------------------------------------------===//
// Serialization
//===----------------------------------------------------------------------===//

static DynamicSliceFusionThunkProto::OffsetProto OffsetToProto(
    const DynamicSliceFusion::Offset& offset) {
  DynamicSliceFusionThunkProto::OffsetProto proto;
  if (auto* c = std::get_if<DynamicSliceFusion::ConstantOffset>(&offset)) {
    proto.set_dimension_number(c->dimension_number);
    proto.set_constant_offset(c->offset);
  } else {
    auto& r = std::get<DynamicSliceFusion::RuntimeOffset>(offset);
    proto.set_dimension_number(r.dimension_number);
    proto.set_runtime_parameter_number(r.parameter_number);
  }
  return proto;
}

static DynamicSliceFusion::Offset OffsetFromProto(
    const DynamicSliceFusionThunkProto::OffsetProto& proto) {
  if (proto.has_constant_offset()) {
    return DynamicSliceFusion::ConstantOffset{proto.constant_offset(),
                                              proto.dimension_number()};
  }
  return DynamicSliceFusion::RuntimeOffset{proto.runtime_parameter_number(),
                                           proto.dimension_number()};
}

absl::StatusOr<ThunkProto> DynamicSliceFusionV2Thunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  auto* dsf = proto.mutable_dynamic_slice_fusion_thunk();

  for (const auto& param : parameters_) {
    auto* p = dsf->add_parameters();
    p->set_parameter_number(param.parameter_number);
    *p->mutable_parameter_shape() = param.parameter_shape.ToProto();
    *p->mutable_slice_shape() = param.slice_shape.ToProto();
    if (param.slice_config.has_value()) {
      *p->mutable_slice_config() = *param.slice_config;
    }
    if (param.slice_offsets.has_value()) {
      for (const auto& offset : *param.slice_offsets) {
        *p->add_slice_offsets() = OffsetToProto(offset);
      }
    }
  }

  for (const auto& result : results_) {
    auto* r = dsf->add_results();
    if (result.parameter_number.has_value()) {
      r->set_parameter_number(*result.parameter_number);
    }
    r->set_result_number(result.result_number);
    *r->mutable_result_shape() = result.result_shape.ToProto();
    *r->mutable_update_shape() = result.update_shape.ToProto();
    if (result.update_config.has_value()) {
      *r->mutable_update_config() = *result.update_config;
    }
    if (result.update_offsets.has_value()) {
      for (const auto& offset : *result.update_offsets) {
        *r->add_update_offsets() = OffsetToProto(offset);
      }
    }
  }

  for (const auto& buf : parameter_buffers_) {
    ASSIGN_OR_RETURN(*dsf->add_parameter_buffers(), buf.ToProto());
  }

  for (const auto& buf : result_buffers_) {
    ASSIGN_OR_RETURN(*dsf->add_result_buffers(), buf.ToProto());
  }

  for (const auto& alloc : slice_allocations_) {
    *dsf->add_slice_allocations() = alloc.ToProto();
  }

  for (const auto& thunk : executor_.thunks()) {
    ASSIGN_OR_RETURN(*dsf->mutable_embedded_thunks()->add_thunks(),
                     thunk->ToProto());
  }

  dsf->set_verify_offsets(verify_offsets_);

  return proto;
}

absl::StatusOr<std::unique_ptr<DynamicSliceFusionV2Thunk>>
DynamicSliceFusionV2Thunk::FromProto(
    ThunkInfo thunk_info, const DynamicSliceFusionThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const DeserializerWithCustomAllocations& deserializer) {
  std::vector<DynamicSliceFusion::Parameter> parameters;
  parameters.reserve(proto.parameters().size());
  for (const auto& p : proto.parameters()) {
    std::optional<DynamicSliceConfig> config;
    if (p.has_slice_config()) {
      config = p.slice_config();
    }
    ASSIGN_OR_RETURN(Shape parameter_shape,
                     Shape::FromProto(p.parameter_shape()));
    ASSIGN_OR_RETURN(Shape slice_shape, Shape::FromProto(p.slice_shape()));
    std::optional<std::vector<DynamicSliceFusion::Offset>> slice_offsets;
    if (!p.slice_offsets().empty()) {
      slice_offsets.emplace();
      for (const auto& o : p.slice_offsets()) {
        slice_offsets->push_back(OffsetFromProto(o));
      }
    }
    parameters.push_back(DynamicSliceFusion::Parameter{
        p.parameter_number(),
        std::move(parameter_shape),
        std::move(slice_shape),
        config,
        std::move(slice_offsets),
    });
  }

  std::vector<DynamicSliceFusion::Result> results;
  results.reserve(proto.results().size());
  for (const auto& r : proto.results()) {
    std::optional<DynamicSliceConfig> update_config;
    if (r.has_update_config()) {
      update_config = r.update_config();
    }
    ASSIGN_OR_RETURN(Shape result_shape, Shape::FromProto(r.result_shape()));
    ASSIGN_OR_RETURN(Shape update_shape, Shape::FromProto(r.update_shape()));
    std::optional<std::vector<DynamicSliceFusion::Offset>> update_offsets;
    if (!r.update_offsets().empty()) {
      update_offsets.emplace();
      for (const auto& o : r.update_offsets()) {
        update_offsets->push_back(OffsetFromProto(o));
      }
    }
    results.push_back(DynamicSliceFusion::Result{
        r.has_parameter_number() ? std::optional<int64_t>(r.parameter_number())
                                 : std::nullopt,
        r.result_number(),
        std::move(result_shape),
        std::move(update_shape),
        update_config,
        std::move(update_offsets),
    });
  }

  std::vector<BufferAllocation::Slice> parameter_buffers;
  parameter_buffers.reserve(proto.parameter_buffers().size());
  for (const auto& buf_proto : proto.parameter_buffers()) {
    ASSIGN_OR_RETURN(auto slice, BufferAllocation::Slice::FromProto(
                                     buf_proto, buffer_allocations));
    parameter_buffers.push_back(slice);
  }

  std::vector<BufferAllocation::Slice> result_buffers;
  result_buffers.reserve(proto.result_buffers().size());
  for (const auto& buf_proto : proto.result_buffers()) {
    ASSIGN_OR_RETURN(auto slice, BufferAllocation::Slice::FromProto(
                                     buf_proto, buffer_allocations));
    result_buffers.push_back(slice);
  }

  std::vector<BufferAllocation> slice_allocations;
  slice_allocations.reserve(proto.slice_allocations().size());
  for (const auto& alloc_proto : proto.slice_allocations()) {
    slice_allocations.push_back(BufferAllocation::FromProto(alloc_proto));
  }

  ThunkSequence embedded_thunks;
  embedded_thunks.reserve(proto.embedded_thunks().thunks().size());
  for (const auto& thunk_proto : proto.embedded_thunks().thunks()) {
    ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk,
                     deserializer(thunk_proto, slice_allocations));
    embedded_thunks.push_back(std::move(thunk));
  }

  return std::make_unique<DynamicSliceFusionV2Thunk>(
      std::move(thunk_info), std::move(parameters), std::move(results),
      std::move(parameter_buffers), std::move(result_buffers),
      std::move(slice_allocations), std::move(embedded_thunks),
      proto.verify_offsets());
}

}  // namespace xla::gpu
