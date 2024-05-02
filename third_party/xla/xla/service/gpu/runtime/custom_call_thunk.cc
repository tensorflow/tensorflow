/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/custom_call_thunk.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

using xla::ffi::CallFrame;
using xla::ffi::CallFrameBuilder;
using xla::ffi::CallOptions;

CustomCallThunk::CustomCallThunk(ThunkInfo thunk_info,
                                 CustomCallTarget call_target,
                                 std::vector<std::optional<Slice>> operands,
                                 std::vector<std::optional<Slice>> results,
                                 const std::string& opaque)
    : Thunk(Thunk::kCustomCall, thunk_info),
      operands_(std::move(operands)),
      results_(std::move(results)),
      call_target_(std::move(call_target)),
      opaque_(opaque) {}

CustomCallThunk::CustomCallThunk(ThunkInfo thunk_info, XLA_FFI_Handler* handler,
                                 std::vector<std::optional<Slice>> operands,
                                 std::vector<std::optional<Slice>> results,
                                 AttributesMap attributes,
                                 const HloComputation* called_computation)
    : Thunk(Thunk::kCustomCall, thunk_info),
      operands_(std::move(operands)),
      results_(std::move(results)),
      handler_(std::move(handler)),
      attributes_(std::move(attributes)),
      called_computation_(called_computation) {}

absl::Status CustomCallThunk::ExecuteCustomCall(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<Slice>& slice : slices) {
      if (!slice.has_value()) {
        buffers.push_back(nullptr);
        continue;
      }

      if (!slice->slice.allocation())
        return Internal("custom call input missing buffer allocation");

      buffers.push_back(
          params.buffer_allocations->GetDeviceAddress(slice->slice).opaque());
    }
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  auto gpu_stream = se::gpu::AsGpuStreamValue(params.stream);
  XlaCustomCallStatus custom_call_status;
  call_target_(gpu_stream, buffers.data(), opaque_.data(), opaque_.size(),
               &custom_call_status);
  auto message = CustomCallStatusGetMessage(&custom_call_status);
  if (message) {
    return Internal("CustomCall failed: %s", *message);
  } else {
    return absl::OkStatus();
  }
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Custom calls on GPU are not supported in this configuration. Please "
      "build with --config=cuda or --config=rocm");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

absl::Status CustomCallThunk::ExecuteFfiHandler(const ExecuteParams& params) {
  // TODO(ezhulenev): This is not the most optimal approach, as we'll be doing
  // a lot of extra allocation on every call. We have to keep attributes
  // separate from arguments, as they do not change after thunk is constructed.
  CallFrameBuilder builder;

  for (auto& operand : operands_) {
    if (!operand.has_value())
      return Internal("FFI handlers do not support tokens (yet)!");
    if (!operand->slice.allocation())
      return Internal("custom call argument missing buffer allocation");

    builder.AddBufferArg(
        params.buffer_allocations->GetDeviceAddress(operand->slice),
        operand->shape.element_type(), operand->shape.dimensions());
  }

  for (auto& result : results_) {
    if (!result.has_value())
      return Internal("FFI handlers do not support tokens (yet)!");
    if (!result->slice.allocation())
      return Internal("custom call result missing buffer allocation");

    builder.AddBufferRet(
        params.buffer_allocations->GetDeviceAddress(result->slice),
        result->shape.element_type(), result->shape.dimensions());
  }

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Append(attributes_);

  builder.AddAttributes(attrs.Build());
  CallFrame call_frame = builder.Build();

  // TODO(ezhulenev): Remove `ServiceExecutableRunOptions` from FFI handler
  // execution context, as apparently it's not easily accessible from Thunk.
  ExecutableRunOptions run_options;
  run_options.set_stream(params.stream);
  run_options.set_allocator(params.buffer_allocations->memory_allocator());
  run_options.set_device_ordinal(params.buffer_allocations->device_ordinal());
  ServiceExecutableRunOptions service_run_options(run_options);

  CallOptions options = {&service_run_options, called_computation_};
  return Call(handler_, call_frame, options);
}

absl::Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  return handler_ ? ExecuteFfiHandler(params) : ExecuteCustomCall(params);
}

absl::StatusOr<CustomCallThunk::AttributesMap> BuildAttributesMap(
    mlir::DictionaryAttr dict) {
  CustomCallThunk::AttributesMap attributes;
  for (auto& kv : dict) {
    std::string_view name = kv.getName().strref();

    auto boolean = [&](mlir::BoolAttr boolean) {
      attributes[name] = static_cast<bool>(boolean.getValue());
      return absl::OkStatus();
    };

    auto integer = [&](mlir::IntegerAttr integer) {
      switch (integer.getType().getIntOrFloatBitWidth()) {
        case 1:
          attributes[name] = static_cast<bool>(integer.getInt());
          return absl::OkStatus();
        case 8:
          attributes[name] = static_cast<int8_t>(integer.getInt());
          return absl::OkStatus();
        case 16:
          attributes[name] = static_cast<int16_t>(integer.getInt());
          return absl::OkStatus();
        case 32:
          attributes[name] = static_cast<int32_t>(integer.getInt());
          return absl::OkStatus();
        case 64:
          attributes[name] = static_cast<int64_t>(integer.getInt());
          return absl::OkStatus();
        default:
          return absl::InvalidArgumentError(absl::StrCat(
              "Unsupported integer attribute bit width for attribute: ", name));
      }
    };

    auto fp = [&](mlir::FloatAttr fp) {
      switch (fp.getType().getIntOrFloatBitWidth()) {
        case 32:
          attributes[name] = static_cast<float>(fp.getValue().convertToFloat());
          return absl::OkStatus();
        case 64:
          attributes[name] =
              static_cast<double>(fp.getValue().convertToDouble());
          return absl::OkStatus();
        default:
          return absl::InvalidArgumentError(absl::StrCat(
              "Unsupported float attribute bit width for attribute: ", name));
      }
    };

    auto arr = [&](mlir::DenseArrayAttr arr) {
      if (auto dense = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseI16ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseF32ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseF64ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported array element type for attribute: ", name));
      }
    };

    auto str = [&](mlir::StringAttr str) {
      attributes[name] = str.getValue().str();
      return absl::OkStatus();
    };

    TF_RETURN_IF_ERROR(
        llvm::TypeSwitch<mlir::Attribute, Status>(kv.getValue())
            .Case<mlir::BoolAttr>(boolean)
            .Case<mlir::IntegerAttr>(integer)
            .Case<mlir::FloatAttr>(fp)
            .Case<mlir::DenseArrayAttr>(arr)
            .Case<mlir::StringAttr>(str)
            .Default([&](mlir::Attribute) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Unsupported attribute type for attribute: ", name));
            }));
  }
  return attributes;
}

}  // namespace gpu
}  // namespace xla
