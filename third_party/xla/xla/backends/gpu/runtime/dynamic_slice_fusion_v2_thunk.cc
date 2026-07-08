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
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/comparison_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

using Offset = DynamicSliceFusion::Offset;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Computes the raw byte offset from the annotated DynamicSliceConfig:
//   byte_offset + loop_iteration[loop_index] * byte_stride
static int64_t ComputeSliceOffset(const DynamicSliceConfig& config,
                                  absl::Span<const WhileLoopState> loop_nest) {
  int64_t iteration = 0;
  if (config.has_loop_index() && config.loop_index() < loop_nest.size()) {
    iteration =
        loop_nest[loop_nest.size() - 1 - config.loop_index()].loop_iteration;
  }
  return config.byte_offset() + iteration * config.byte_stride();
}

// Computes the raw byte offset from actual offset expressions. Runtime scalar
// parameters are D2H-copied from device before evaluation.
static absl::StatusOr<int64_t> ComputeSliceOffset(
    const Shape& src_shape, absl::Span<const Offset> offsets,
    absl::Span<const std::pair<int64_t, int64_t>> parameters) {
  auto byte_strides = ShapeUtil::ByteStrides(src_shape);
  if (!byte_strides.has_value()) {
    return InvalidArgument("Failed to compute byte strides for shape %s",
                           ShapeUtil::HumanString(src_shape));
  }

  int64_t byte_offset = 0;
  for (const auto& offset : offsets) {
    int64_t dim = offset.dimension_number;
    ASSIGN_OR_RETURN(int64_t idx,
                     DynamicSliceFusion::Evaluate(offset.expr, parameters));
    byte_offset += idx * (*byte_strides)[dim];
  }
  return byte_offset;
}

// Clamps a byte offset to [0, buffer_size - slice_size] matching DUS semantics.
static int64_t ClampSliceOffset(int64_t offset, int64_t buffer_size,
                                int64_t slice_size) {
  return std::clamp(offset, int64_t{0}, buffer_size - slice_size);
}

//===----------------------------------------------------------------------===//
// DynamicSliceFusionV2Thunk
//===----------------------------------------------------------------------===//

DynamicSliceFusionV2Thunk::DynamicSliceFusionV2Thunk(
    ThunkInfo thunk_info, std::vector<DynamicSliceFusion::Parameter> parameters,
    std::vector<DynamicSliceFusion::Result> results,
    std::vector<BufferAllocation::Slice> parameter_buffers,
    std::vector<BufferAllocation::Slice> result_buffers,
    std::vector<BufferAllocation> embedded_allocations,
    ThunkSequence embedded_thunks, bool verify_offsets)
    : Command(Kind::kDynamicSliceFusion, std::move(thunk_info)),
      parameters_(std::move(parameters)),
      results_(std::move(results)),
      parameter_buffers_(std::move(parameter_buffers)),
      result_buffers_(std::move(result_buffers)),
      embedded_allocations_(std::move(embedded_allocations)),
      executor_(std::move(embedded_thunks)),
      verify_offsets_(verify_offsets) {}

static bool IsLoopDependent(const std::optional<DynamicSliceConfig>& config) {
  return config.has_value() && config->has_loop_index() &&
         config->byte_stride() != 0;
}

bool DynamicSliceFusionV2Thunk::HasLoopDependentOffsets() const {
  bool param_loop_dependent = absl::c_any_of(
      parameters_, [](const DynamicSliceFusion::Parameter& parameter) {
        return IsLoopDependent(parameter.slice_config);
      });
  bool result_loop_dependent =
      absl::c_any_of(results_, [](const DynamicSliceFusion::Result& result) {
        return IsLoopDependent(result.update_config);
      });
  return param_loop_dependent || result_loop_dependent;
}

// TODO(shawnwang18): Build command executors for thunks with nested thunk
// executors in their constructors when nested thunks are command-buffer
// compatible.
absl::Status DynamicSliceFusionV2Thunk::SetOrUpdateCommandBufferExecutor(
    CommandExecutor command_executor) {
  command_executor_ = std::move(command_executor);
  return absl::OkStatus();
}

std::string DynamicSliceFusionV2Thunk::ToString(int indent) const {
  std::string result;
  absl::StrAppendFormat(&result, "params=%d, results=%d\n", parameters_.size(),
                        results_.size());
  absl::StrAppend(&result, executor_.thunks().ToString(indent + 1));
  return result;
}

absl::Status DynamicSliceFusionV2Thunk::Prepare(const PrepareParams& params) {
  // Embedded thunks' slices are relative to `embedded_allocations_`, so
  // prepare them with the same fusion-local view that ExecuteOnStream builds.
  std::vector<se::DeviceAddressBase> buffers = BuildDynamicSliceBuffers(
      *params.buffer_allocations, IsInsideWhileLoopNest());
  BufferAllocations embedded_allocs(
      buffers, params.buffer_allocations->device_ordinal(),
      params.buffer_allocations->memory_allocator());
  PrepareParams embedded_params = params;
  embedded_params.buffer_allocations = &embedded_allocs;
  if (command_executor_.has_value()) {
    RETURN_IF_ERROR(command_executor_->Prepare(embedded_params));
  }
  return executor_.Prepare(embedded_params);
}

absl::Status DynamicSliceFusionV2Thunk::Initialize(
    const InitializeParams& params) {
  if (command_executor_.has_value()) {
    RETURN_IF_ERROR(command_executor_->Initialize(params));
  }
  return executor_.Initialize(params);
}

absl::Status DynamicSliceFusionV2Thunk::VerifyBufferAssignment(
    absl::Span<const DynamicSliceFusion::Result> results,
    absl::Span<const BufferAllocation::Slice> parameter_buffers,
    absl::Span<const BufferAllocation::Slice> result_buffers) {
  for (const DynamicSliceFusion::Result& result : results) {
    if (!result.parameter_number.has_value()) {
      continue;
    }

    const int64_t parameter_number = *result.parameter_number;
    const int64_t parameter_buffer_count = parameter_buffers.size();

    if (parameter_number < 0 || parameter_number >= parameter_buffer_count) {
      return Internal(
          "DUS result %d targets missing fusion parameter %d; parameter buffer "
          "count is %d",
          result.result_number, parameter_number, parameter_buffer_count);
    }

    const int64_t result_buffer_count = result_buffers.size();
    if (result.result_number < 0 ||
        result.result_number >= result_buffer_count) {
      return Internal(
          "DUS result %d has no result buffer; result buffer count is %d",
          result.result_number, result_buffer_count);
    }

    const BufferAllocation::Slice& parameter_buffer =
        parameter_buffers[parameter_number];
    const BufferAllocation::Slice& result_buffer =
        result_buffers[result.result_number];
    if (parameter_buffer != result_buffer) {
      return Internal(
          "DUS result %d must alias fusion parameter %d, but result buffer is "
          "%v and parameter buffer is %v",
          result.result_number, parameter_number, result_buffer,
          parameter_buffer);
    }
  }
  return absl::OkStatus();
}

static absl::Status VerifySliceOffset(
    se::Stream& stream, const BufferAllocations& orig, absl::string_view kind,
    size_t idx, const std::optional<DynamicSliceConfig>& config,
    const std::optional<std::vector<Offset>>& offsets, const Shape& src_shape,
    const Shape& dst_shape, absl::Span<const WhileLoopState> loop_nest,
    absl::Span<const BufferAllocation::Slice> parameter_buffers) {
  if (!config.has_value() || !offsets.has_value()) {
    return absl::OkStatus();
  }

  // Collect offset expression leaves that depend on runtime values.
  absl::btree_set<int64_t> runtime_parameters;
  for (const auto& offset : *offsets) {
    for (int64_t parameter_number :
         DynamicSliceFusion::CollectOffsetParameters(offset.expr)) {
      runtime_parameters.insert(parameter_number);
    }
  }

  // If all offsets are static, we have nothing to verify.
  if (runtime_parameters.empty()) {
    return absl::OkStatus();
  }

  // Copy offset values to host. parameter_buffers is indexed by fusion
  // parameter number, so we can index directly.
  std::vector<std::pair<int64_t, int64_t>> parameters;
  parameters.reserve(runtime_parameters.size());
  for (int64_t parameter_number : runtime_parameters) {
    if (parameter_number < 0 || parameter_number >= parameter_buffers.size()) {
      return Internal("Missing offset buffer at parameter %d",
                      parameter_number);
    }

    const BufferAllocation::Slice& parameter_buffer =
        parameter_buffers[parameter_number];
    auto src = orig.GetDeviceAddress(parameter_buffer);
    if (parameter_buffer.size() == sizeof(int32_t)) {
      int32_t value = 0;
      RETURN_IF_ERROR(stream.Memcpy(&value, src, sizeof(int32_t)));
      RETURN_IF_ERROR(stream.BlockHostUntilDone());
      parameters.emplace_back(parameter_number, value);
    } else if (parameter_buffer.size() == sizeof(int64_t)) {
      int64_t value = 0;
      RETURN_IF_ERROR(stream.Memcpy(&value, src, sizeof(int64_t)));
      RETURN_IF_ERROR(stream.BlockHostUntilDone());
      parameters.emplace_back(parameter_number, value);
    } else {
      return Internal(
          "Expected S32- or S64-sized offset buffer at parameter %d, got %d "
          "bytes",
          parameter_number, parameter_buffer.size());
    }
  }

  // Compare offsets after clamping both to [0, buffer_size - slice_size].
  int64_t buffer_size = ShapeUtil::ByteSizeOf(src_shape);
  int64_t slice_size = ShapeUtil::ByteSizeOf(dst_shape);
  ASSIGN_OR_RETURN(int64_t offset_from_exprs,
                   ComputeSliceOffset(src_shape, *offsets, parameters));
  int64_t actual_offset =
      ClampSliceOffset(offset_from_exprs, buffer_size, slice_size);
  int64_t annotated_offset = ClampSliceOffset(
      ComputeSliceOffset(*config, loop_nest), buffer_size, slice_size);

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
        parameters[i].slice_shape, loop_nest, parameter_buffers));
  }

  for (size_t j = 0; j < results.size(); ++j) {
    RETURN_IF_ERROR(VerifySliceOffset(
        stream, orig, "result", j, results[j].update_config,
        results[j].update_offsets, results[j].result_shape,
        results[j].update_shape, loop_nest, parameter_buffers));
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

  BufferAllocations embedded_allocs(buffers, orig.device_ordinal(),
                                    orig.memory_allocator());
  ExecuteParams dynamic_slice_params =
      ExecuteParams::CloneWithNewAllocations(params, embedded_allocs);
  return executor_.ExecuteOnStream(dynamic_slice_params);
}

absl::StatusOr<const se::CommandBuffer::Command*>
DynamicSliceFusionV2Thunk::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  RecordAction record_action,
                                  se::CommandBuffer* command_buffer) {
  if (verify_offsets_) {
    return FailedPrecondition(
        "DynamicSliceFusionV2Thunk command-buffer recording does not support "
        "runtime offset verification");
  }
  if (!command_executor_.has_value()) {
    return FailedPrecondition(
        "DynamicSliceFusionV2Thunk command executor is not initialized");
  }
  if (command_executor_->empty()) {
    return nullptr;
  }

  auto child_record_params = [&]() {
    Command::RecordParams params = record_params;
    params.updated_allocs = std::nullopt;
    return params;
  };

  auto create_params =
      [&](BufferAllocations& embedded_allocs) -> Thunk::ExecuteParams {
    return Thunk::ExecuteParams::CloneWithNewAllocations(execute_params,
                                                         embedded_allocs);
  };

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateChildCommand(
        [&, this](se::CommandBuffer* child_command_buffer) -> absl::Status {
          std::vector<se::DeviceAddressBase> buffers = BuildDynamicSliceBuffers(
              *execute_params.buffer_allocations, IsInsideWhileLoopNest());
          BufferAllocations embedded_allocs(
              buffers, execute_params.buffer_allocations->device_ordinal(),
              execute_params.buffer_allocations->memory_allocator());
          Thunk::ExecuteParams params = create_params(embedded_allocs);
          Command::RecordParams child_params = child_record_params();
          return command_executor_
              ->RecordCreate(params, child_params, child_command_buffer,
                             /*dependencies=*/{})
              .status();
        },
        create->dependencies);
  }

  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(command_buffer->UpdateChildCommand(
        update->command,
        [&, this](se::CommandBuffer* child_command_buffer) -> absl::Status {
          std::vector<se::DeviceAddressBase> buffers = BuildDynamicSliceBuffers(
              *execute_params.buffer_allocations, IsInsideWhileLoopNest());
          BufferAllocations embedded_allocs(
              buffers, execute_params.buffer_allocations->device_ordinal(),
              execute_params.buffer_allocations->memory_allocator());
          Thunk::ExecuteParams params = create_params(embedded_allocs);
          Command::RecordParams child_params = child_record_params();
          return command_executor_->RecordUpdate(params, child_params,
                                                 child_command_buffer);
        }));
    return update->command;
  }

  return Internal("Invalid record action");
}

bool DynamicSliceFusionV2Thunk::requires_update_on_initialize() const {
  return command_executor_.has_value() &&
         command_executor_->requires_update_on_initialize();
}

bool DynamicSliceFusionV2Thunk::requires_update_on_execute() const {
  return HasLoopDependentOffsets() ||
         (command_executor_.has_value() &&
          command_executor_->requires_update_on_execute());
}

bool DynamicSliceFusionV2Thunk::support_loop_unroll() const {
  return !command_executor_.has_value() ||
         command_executor_->support_loop_unroll();
}

std::vector<se::DeviceAddressBase>
DynamicSliceFusionV2Thunk::BuildDynamicSliceBuffers(
    const BufferAllocations& orig,
    absl::Span<const WhileLoopState> loop_nest) const {
  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(parameters_.size() + results_.size());

  for (size_t i = 0; i < parameters_.size(); ++i) {
    se::DeviceAddressBase addr = orig.GetDeviceAddress(
        parameter_buffers_[parameters_[i].parameter_number]);
    if (parameters_[i].slice_config.has_value()) {
      int64_t sliced_size = embedded_allocations_[i].size();
      int64_t offset =
          ComputeSliceOffset(*parameters_[i].slice_config, loop_nest);
      int64_t clamped_offset =
          ClampSliceOffset(offset, addr.size(), sliced_size);
      XLA_VLOG_DEVICE(3, orig.device_ordinal()) << absl::StrFormat(
          "  param[%d]: base=%p size=%d -> offset=%d clamped=%d sliced_size=%d",
          i, addr.opaque(), addr.size(), offset, clamped_offset, sliced_size);
      addr = addr.GetByteSlice(clamped_offset, sliced_size);
    }
    buffers.push_back(addr);
  }

  for (size_t j = 0; j < results_.size(); ++j) {
    se::DeviceAddressBase addr =
        orig.GetDeviceAddress(result_buffers_[results_[j].result_number]);
    if (results_[j].update_config.has_value()) {
      int64_t sliced_size =
          embedded_allocations_[parameters_.size() + j].size();
      int64_t offset =
          ComputeSliceOffset(*results_[j].update_config, loop_nest);
      int64_t clamped_offset =
          ClampSliceOffset(offset, addr.size(), sliced_size);
      XLA_VLOG_DEVICE(3, orig.device_ordinal()) << absl::StrFormat(
          "  result[%d]: base=%p size=%d -> offset=%d clamped=%d "
          "sliced_size=%d",
          j, addr.opaque(), addr.size(), offset, clamped_offset, sliced_size);
      addr = addr.GetByteSlice(clamped_offset, sliced_size);
    }
    buffers.push_back(addr);
  }

  return buffers;
}

Thunk::BufferUses DynamicSliceFusionV2Thunk::buffer_uses() const {
  BufferUses uses;
  for (size_t i = 0; i < parameters_.size(); ++i) {
    uses.push_back(
        BufferUse::Read(parameter_buffers_[parameters_[i].parameter_number],
                        parameters_[i].parameter_shape));
  }
  for (size_t j = 0; j < results_.size(); ++j) {
    uses.push_back(BufferUse::Write(result_buffers_[results_[j].result_number],
                                    results_[j].result_shape));
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

using OffsetExprProto = DynamicSliceFusionThunkProto::OffsetExprProto;

static OffsetExprProto::Kind OffsetExprKindToProto(const Offset::Expr& expr) {
  return std::visit(
      [](const auto& e) -> OffsetExprProto::Kind {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, Offset::Expr::Constant>) {
          return OffsetExprProto::CONSTANT;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Parameter>) {
          return OffsetExprProto::PARAMETER;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Add>) {
          return OffsetExprProto::ADD;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Subtract>) {
          return OffsetExprProto::SUBTRACT;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Multiply>) {
          return OffsetExprProto::MULTIPLY;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Compare>) {
          return OffsetExprProto::COMPARE;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Select>) {
          return OffsetExprProto::SELECT;
        } else {
          return OffsetExprProto::KIND_UNKNOWN;
        }
      },
      expr.value);
}

static OffsetExprProto::CompareDirection CompareDirectionToProto(
    ComparisonDirection direction) {
  switch (direction) {
    case ComparisonDirection::kEq:
      return OffsetExprProto::EQ;
    case ComparisonDirection::kNe:
      return OffsetExprProto::NE;
    case ComparisonDirection::kGe:
      return OffsetExprProto::GE;
    case ComparisonDirection::kGt:
      return OffsetExprProto::GT;
    case ComparisonDirection::kLe:
      return OffsetExprProto::LE;
    case ComparisonDirection::kLt:
      return OffsetExprProto::LT;
  }
  return OffsetExprProto::COMPARE_DIRECTION_UNKNOWN;
}

static absl::StatusOr<ComparisonDirection> CompareDirectionFromProto(
    OffsetExprProto::CompareDirection direction) {
  switch (direction) {
    case OffsetExprProto::EQ:
      return ComparisonDirection::kEq;
    case OffsetExprProto::NE:
      return ComparisonDirection::kNe;
    case OffsetExprProto::GE:
      return ComparisonDirection::kGe;
    case OffsetExprProto::GT:
      return ComparisonDirection::kGt;
    case OffsetExprProto::LE:
      return ComparisonDirection::kLe;
    case OffsetExprProto::LT:
      return ComparisonDirection::kLt;
    case OffsetExprProto::COMPARE_DIRECTION_UNKNOWN:
      return InvalidArgument("Unknown offset expression comparison direction");
    default:
      return InvalidArgument("Unknown offset expression comparison direction");
  }
}

static OffsetExprProto OffsetExprToProto(const Offset::Expr& expr) {
  OffsetExprProto proto;
  proto.set_kind(OffsetExprKindToProto(expr));
  std::visit(
      [&](const auto& e) {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, Offset::Expr::Constant>) {
          proto.set_value(e.value);
        } else if constexpr (std::is_same_v<T, Offset::Expr::Parameter>) {
          proto.set_value(e.parameter_number);
        } else {
          if constexpr (std::is_same_v<T, Offset::Expr::Compare>) {
            proto.set_compare_direction(CompareDirectionToProto(e.direction));
          }
          for (const Offset::Expr& arg : e.args) {
            *proto.add_operands() = OffsetExprToProto(arg);
          }
        }
      },
      expr.value);
  return proto;
}

static absl::Status VerifyOperandCount(const OffsetExprProto& proto,
                                       size_t expected) {
  if (proto.operands().size() != expected) {
    return InvalidArgument(
        "Expected offset expression proto to have %d "
        "operands, got %d",
        expected, proto.operands().size());
  }
  return absl::OkStatus();
}

static absl::StatusOr<Offset::Expr> OffsetExprFromProto(
    const OffsetExprProto& proto) {
  std::vector<Offset::Expr> args;
  args.reserve(proto.operands().size());
  for (const OffsetExprProto& operand_proto : proto.operands()) {
    ASSIGN_OR_RETURN(Offset::Expr operand, OffsetExprFromProto(operand_proto));
    args.push_back(std::move(operand));
  }

  switch (proto.kind()) {
    case OffsetExprProto::CONSTANT:
      RETURN_IF_ERROR(VerifyOperandCount(proto, 0));
      return Offset::Constant(proto.value());
    case OffsetExprProto::PARAMETER:
      RETURN_IF_ERROR(VerifyOperandCount(proto, 0));
      return Offset::Parameter(proto.value());
    case OffsetExprProto::ADD:
      RETURN_IF_ERROR(VerifyOperandCount(proto, 2));
      return Offset::Add(std::move(args[0]), std::move(args[1]));
    case OffsetExprProto::SUBTRACT:
      RETURN_IF_ERROR(VerifyOperandCount(proto, 2));
      return Offset::Subtract(std::move(args[0]), std::move(args[1]));
    case OffsetExprProto::MULTIPLY:
      RETURN_IF_ERROR(VerifyOperandCount(proto, 2));
      return Offset::Multiply(std::move(args[0]), std::move(args[1]));
    case OffsetExprProto::COMPARE: {
      RETURN_IF_ERROR(VerifyOperandCount(proto, 2));
      ASSIGN_OR_RETURN(ComparisonDirection direction,
                       CompareDirectionFromProto(proto.compare_direction()));
      return Offset::Compare(direction, std::move(args[0]), std::move(args[1]));
    }
    case OffsetExprProto::SELECT:
      RETURN_IF_ERROR(VerifyOperandCount(proto, 3));
      return Offset::Select(std::move(args[0]), std::move(args[1]),
                            std::move(args[2]));
    case OffsetExprProto::KIND_UNKNOWN:
      return InvalidArgument("Unknown offset expression kind");
    default:
      return InvalidArgument("Unknown offset expression kind");
  }
}

static DynamicSliceFusionThunkProto::OffsetProto OffsetToProto(
    const Offset& offset) {
  DynamicSliceFusionThunkProto::OffsetProto proto;
  proto.set_dimension_number(offset.dimension_number);
  *proto.mutable_offset() = OffsetExprToProto(offset.expr);
  return proto;
}

static absl::StatusOr<Offset> OffsetFromProto(
    const DynamicSliceFusionThunkProto::OffsetProto& proto) {
  if (!proto.has_offset()) {
    return InvalidArgument("Offset proto has no value");
  }
  ASSIGN_OR_RETURN(Offset::Expr expr, OffsetExprFromProto(proto.offset()));
  return Offset{proto.dimension_number(), std::move(expr)};
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

  for (const auto& alloc : embedded_allocations_) {
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
    std::optional<std::vector<Offset>> slice_offsets;
    if (!p.slice_offsets().empty()) {
      slice_offsets.emplace();
      for (const auto& o : p.slice_offsets()) {
        ASSIGN_OR_RETURN(Offset offset, OffsetFromProto(o));
        slice_offsets->push_back(std::move(offset));
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
    std::optional<std::vector<Offset>> update_offsets;
    if (!r.update_offsets().empty()) {
      update_offsets.emplace();
      for (const auto& o : r.update_offsets()) {
        ASSIGN_OR_RETURN(Offset offset, OffsetFromProto(o));
        update_offsets->push_back(std::move(offset));
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

  std::vector<BufferAllocation> embedded_allocations;
  embedded_allocations.reserve(proto.slice_allocations().size());
  for (const auto& alloc_proto : proto.slice_allocations()) {
    embedded_allocations.push_back(BufferAllocation::FromProto(alloc_proto));
  }

  ThunkSequence embedded_thunks;
  embedded_thunks.reserve(proto.embedded_thunks().thunks().size());
  for (const auto& thunk_proto : proto.embedded_thunks().thunks()) {
    ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk,
                     deserializer(thunk_proto, embedded_allocations));
    embedded_thunks.push_back(std::move(thunk));
  }

  return std::make_unique<DynamicSliceFusionV2Thunk>(
      std::move(thunk_info), std::move(parameters), std::move(results),
      std::move(parameter_buffers), std::move(result_buffers),
      std::move(embedded_allocations), std::move(embedded_thunks),
      proto.verify_offsets());
}

}  // namespace xla::gpu
