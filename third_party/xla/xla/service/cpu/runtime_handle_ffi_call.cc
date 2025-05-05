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

#include "xla/service/cpu/runtime_handle_ffi_call.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"
#include "xla/service/custom_call_status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace ffi = xla::ffi;

namespace {

absl::Span<const int64_t> DecodeDims(int64_t* encoded_dims_data) {
  // Annotate memory coming from jit compiled function as initialized to
  // suppress false positives from msan sanitizer.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(encoded_dims_data, sizeof(int64_t));
  auto dims_count = encoded_dims_data[0];
  auto dims_begin = encoded_dims_data + 1;
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(dims_begin, dims_count * sizeof(int64_t));
  return absl::MakeSpan(dims_begin, dims_begin + dims_count);
}

void BuildArgBuffers(absl::Span<const int32_t> types, int64_t* encoded_dims,
                     absl::Span<void* const> address_space,
                     ffi::CallFrameBuilder& builder) {
  int64_t dim_pos = 0;
  for (int64_t i = 0; i < types.size(); ++i) {
    auto dtype = static_cast<xla::PrimitiveType>(types[i]);
    if (dtype == xla::PrimitiveType::TOKEN) {
      builder.AddTokenArg();
      continue;
    }
    auto dims = DecodeDims(encoded_dims + dim_pos);
    auto elem_count = absl::c_accumulate(dims, 1, std::multiplies<int64_t>());
    auto data_width = xla::primitive_util::ByteWidth(dtype) * elem_count;

    builder.AddBufferArg(
        tensorflow::se::DeviceMemoryBase(address_space[i], data_width),
        /*type = */ dtype,
        /*dims = */ dims);
    dim_pos += 1;            // Jumps over count value
    dim_pos += dims.size();  // Jumps over all dimensions in a shape
  }
}

void BuildRetBuffers(absl::Span<const int32_t> types, int64_t* encoded_dims,
                     absl::Span<void* const> address_space,
                     ffi::CallFrameBuilder& builder) {
  int64_t dim_pos = 0;
  for (int64_t i = 0; i < types.size(); ++i) {
    auto dtype = static_cast<xla::PrimitiveType>(types[i]);
    if (dtype == xla::PrimitiveType::TOKEN) {
      builder.AddTokenRet();
      continue;
    }
    auto dims = DecodeDims(encoded_dims + dim_pos);
    auto elem_count = absl::c_accumulate(dims, 1, std::multiplies<int64_t>());
    auto data_width = xla::primitive_util::ByteWidth(dtype) * elem_count;

    builder.AddBufferRet(
        tensorflow::se::DeviceMemoryBase(address_space[i], data_width),
        /*type = */ dtype,
        /*dims = */ dims);
    dim_pos += 1;            // Jumps over count value
    dim_pos += dims.size();  // Jumps over all dimensions in a shape
  }
}

static absl::Status BuildAndCallFfi(
    const xla::ExecutableRunOptions* run_options, absl::string_view target_name,
    absl::string_view backend_config, absl::Span<void* const> outputs,
    absl::Span<void* const> inputs, absl::Span<const int32_t> result_types,
    int64_t* result_dims, absl::Span<const int32_t> operand_types,
    int64_t* operand_dims) {
  CHECK_EQ(outputs.size(), result_types.size());
  CHECK_EQ(inputs.size(), operand_types.size());

  // Find the registered FFI handler for this custom call target.
  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(target_name, "Host");

  if (!registration.ok()) {
    return absl::NotFoundError(
        absl::StrCat("No registered implementation for custom call to ",
                     target_name, " for Host."));
  }

  // For FFI handlers backend config must be a compatible MLIR dictionary.
  ffi::CallFrameBuilder::AttributesMap attributes;
  if (!backend_config.empty() && backend_config != "{}") {
    // Backend config not empty, so proceed to parse it into an MLIR attribute
    // and build an MLIR compatible map of attributes out of it.
    mlir::MLIRContext mlir_context;
    mlir::Attribute attr = mlir::parseAttribute(backend_config, &mlir_context);
    if (auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr)) {
      TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    } else {
      return absl::InternalError(
          "Unsupported backend config. Expected a string parsable into "
          "dictionary attribute");
    }
  }

  auto execution_state = std::make_unique<ffi::ExecutionState>();

  // Initialize handler execution state
  if (registration->bundle.instantiate) {
    ffi::CallFrameBuilder builder(0, 0);
    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(attributes);
    ffi::CallFrameBuilder::AttributesMap attrs_map = attrs.Build();
    builder.AddAttributes(attrs_map);
    ffi::CallFrame call_frame = builder.Build();
    ffi::CallOptions options;
    options.execution_state = execution_state.get();
    TF_RETURN_IF_ERROR(Call(registration->bundle.instantiate, call_frame,
                            options, XLA_FFI_ExecutionStage_INSTANTIATE));
  }

  ffi::CallFrameBuilder builder(inputs.size(), outputs.size());

  // Forward the constructed attributes to the call frame
  ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Append(std::move(attributes));
  builder.AddAttributes(attrs.Build());

  // Decode dimensions metadata into shapes and build operand & result buffers
  BuildArgBuffers(operand_types, operand_dims, inputs, builder);
  BuildRetBuffers(result_types, result_dims, outputs, builder);

  // Forward executable run options to the FFI handlers via the call options.
  ffi::CallOptions call_options = {
      run_options->run_id(),
      run_options->device_ordinal(),
      ffi::CallOptions::CpuOptions{run_options->intra_op_thread_pool()},
      /*called_computation=*/nullptr,
      run_options->ffi_execution_context(),
      execution_state.get()};

  ffi::CallFrame call_frame = builder.Build();
  return ffi::Call(registration->bundle.execute, call_frame, call_options);
}

}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_HandleFfiCall(
    const void* run_options_ptr, const char* target_name_ptr,
    int64_t target_name_len, void** outputs, void** inputs,
    const char* opaque_str_ptr, int64_t opaque_str_len, void* status_opaque,
    int32_t* operand_types, int64_t operand_count, int64_t* operand_dims,
    int32_t* result_types, int64_t result_count, int64_t* result_dims) {
  auto target_name = absl::string_view(target_name_ptr, target_name_len);
  auto backend_config = absl::string_view(opaque_str_ptr, opaque_str_len);
  auto xla_status = reinterpret_cast<XlaCustomCallStatus*>(status_opaque);

  // Annotate memory coming from jit compiled function as initialized to
  // suppress false positives from msan sanitizer.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(result_types,
                                      result_count * sizeof(int32_t));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(operand_types,
                                      operand_count * sizeof(int32_t));

  const xla::ExecutableRunOptions* run_options =
      reinterpret_cast<const xla::ExecutableRunOptions*>(run_options_ptr);

  absl::Status status = BuildAndCallFfi(
      run_options, target_name, backend_config,
      absl::MakeSpan(outputs, result_count),
      absl::MakeSpan(inputs, operand_count),
      absl::MakeSpan(result_types, result_count), result_dims,
      absl::MakeSpan(operand_types, operand_count), operand_dims);

  if (!status.ok()) {
    // In the future, status propagation will likely be possible.
    // However, currently this has to pass through XlaCustomCallStatus
    // which lacks functionality for status codes (it is fixed on INTERNAL)
    XlaCustomCallStatusSetFailure(xla_status, status.message().data(),
                                  status.message().size());
  }
}
