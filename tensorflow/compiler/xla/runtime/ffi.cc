/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/runtime/ffi.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/ffi/ffi_c_api.h"

//===----------------------------------------------------------------------===//
// Define structs forward-declared by XLA FFI C API.
//===----------------------------------------------------------------------===//

struct XLA_FFI_Error {
  XLA_FFI_Error_Code errc;
  std::string error;
};

//===----------------------------------------------------------------------===//

namespace xla {
namespace runtime {
namespace ffi {

// All FFI functions registered in a static dynamic custom call registry.
DynamicCustomCallRegistry& FfiCustomCalls() {
  static auto* registry = new DynamicCustomCallRegistry;
  return *registry;
}

static std::string StructSizeErrorMsg(absl::string_view struct_name,
                                      size_t expected_size,
                                      size_t actual_size) {
  return absl::StrCat("Unexpected ", struct_name, " size: expected ",
                      expected_size, ", got ", actual_size,
                      ". Check installed software versions.");
}

static absl::Status CheckMatchingStructSizes(absl::string_view struct_name,
                                             size_t expected_size,
                                             size_t actual_size) {
  if (expected_size != actual_size) {
    return absl::InvalidArgumentError(
        StructSizeErrorMsg(struct_name, expected_size, actual_size));
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Implement XLA FFI error reporting API.
//===----------------------------------------------------------------------===//

static XLA_FFI_Error* CreateError(XLA_FFI_Error_Create_Args* args) {
  absl::Status struct_size_check = CheckMatchingStructSizes(
      "XLA_FFI_Error_Create_Args", XLA_FFI_Error_Create_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) LOG(ERROR) << struct_size_check.message();

  return new XLA_FFI_Error{args->errc, std::string(args->message)};
}

//===----------------------------------------------------------------------===//
// Adaptor from the Xla custom call to an Xla FFI calling convention.
//===----------------------------------------------------------------------===//

absl::StatusCode ConvertErrorCode(XLA_FFI_Error_Code errc) {
  switch (errc) {
    case XLA_FFI_Error_Code_ABORTED:
      return absl::StatusCode::kAborted;
    case XLA_FFI_Error_Code_CANCELLED:
      return absl::StatusCode::kCancelled;
    case XLA_FFI_Error_Code_UNKNOWN:
      return absl::StatusCode::kUnknown;
    case XLA_FFI_Error_Code_INVALID_ARGUMENT:
      return absl::StatusCode::kInvalidArgument;
    case XLA_FFI_Error_Code_DEADLINE_EXCEEDED:
      return absl::StatusCode::kDeadlineExceeded;
    case XLA_FFI_Error_Code_NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case XLA_FFI_Error_Code_ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case XLA_FFI_Error_Code_PERMISSION_DENIED:
      return absl::StatusCode::kPermissionDenied;
    case XLA_FFI_Error_Code_RESOURCE_EXHAUSTED:
      return absl::StatusCode::kResourceExhausted;
    case XLA_FFI_Error_Code_FAILED_PRECONDITION:
      return absl::StatusCode::kFailedPrecondition;
    case XLA_FFI_Error_Code_OUT_OF_RANGE:
      return absl::StatusCode::kOutOfRange;
    case XLA_FFI_Error_Code_UNIMPLEMENTED:
      return absl::StatusCode::kUnimplemented;
    case XLA_FFI_Error_Code_INTERNAL:
      return absl::StatusCode::kInternal;
    case XLA_FFI_Error_Code_UNAVAILABLE:
      return absl::StatusCode::kUnavailable;
    case XLA_FFI_Error_Code_DATA_LOSS:
      return absl::StatusCode::kDataLoss;
    case XLA_FFI_Error_Code_UNAUTHENTICATED:
      return absl::StatusCode::kUnauthenticated;
    default:
      return absl::StatusCode::kUnknown;
  }
}

template <typename T>
static XLA_FFI_TypeId FfiTypeId() {
  return TypeID::get<Tagged<T>>().getAsOpaquePointer();
}

class FfiCustomCall : public CustomCall {
 public:
  FfiCustomCall(std::string_view name, XLA_FFI_Function function)
      : name_(name), function_(function) {}

  std::string_view name() const final { return name_; }

  LogicalResult call(void** args, void** attrs, void** rets,
                     const UserData* user_data,
                     const DiagnosticEngine* diagnostic) const final {
    // Prepare FFI execution context.
    XLA_FFI_ExecutionContext ctx;
    ctx.XLA_FFI_Error_Create = CreateError;
    // Scalar type ids.
    ctx.XLA_FFI_Get_Float_TypeId = FfiTypeId<float>;
    ctx.XLA_FFI_Get_Int32_TypeId = FfiTypeId<int32_t>;
    // Buffer type ids (we call them memrefs in Xla custom calls).
    ctx.XLA_FFI_Get_BufferArg_TypeId = FfiTypeId<MemrefView>;
    ctx.XLA_FFI_Get_StridedBufferArg_TypeId = FfiTypeId<StridedMemrefView>;

    XLA_FFI_Function_Args ffi_args;
    ffi_args.struct_size = XLA_FFI_Function_Args_STRUCT_SIZE;
    ffi_args.priv = nullptr;
    ffi_args.ctx = &ctx;
    ffi_args.args = args;
    ffi_args.attrs = attrs;
    ffi_args.rets = rets;

    // Execute FFI handler and maybe report an error.
    if (XLA_FFI_Error* error = function_(&ffi_args)) {
      return diagnostic->EmitError(
          absl::Status(ConvertErrorCode(error->errc), error->error));
    }

    return success();
  }

 private:
  std::string name_;
  XLA_FFI_Function function_;
};

//===----------------------------------------------------------------------===//

static void Register(XLA_FFI_Register_Args* args) {
  absl::Status struct_size_check = CheckMatchingStructSizes(
      "XLA_FFI_Register_Args", XLA_FFI_Register_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) LOG(ERROR) << struct_size_check.message();

  auto& registry = FfiCustomCalls();
  registry.Register(
      std::make_unique<FfiCustomCall>(args->target, args->function));
}

}  // namespace ffi
}  // namespace runtime
}  // namespace xla

const XLA_FFI_Api ffi_api = {
    ::xla::runtime::ffi::Register,
};

const XLA_FFI_Api* GetXlaFfiApi() { return &ffi_api; }
