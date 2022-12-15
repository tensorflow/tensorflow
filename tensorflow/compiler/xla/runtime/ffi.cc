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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/ffi/ffi_c_api.h"
#include "tensorflow/compiler/xla/runtime/module.h"

//===----------------------------------------------------------------------===//
// Define structs forward-declared by XLA FFI C API.
//===----------------------------------------------------------------------===//

struct XLA_FFI_Error {
  XLA_FFI_Error_Code errc;
  std::string error;
};

struct XLA_FFI_ExecutionContext {
  XLA_FFI_Module_State* state;
  XLA_FFI_Stream* stream;
};

//===----------------------------------------------------------------------===//

namespace xla {
namespace runtime {
namespace ffi {

//===----------------------------------------------------------------------===//
// Helper functions to check ABI compatibility.
//===----------------------------------------------------------------------===//

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
// Error code conversion helper.
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

//===----------------------------------------------------------------------===//
// Adaptor from the Xla custom call to an Xla FFI calling convention.
//===----------------------------------------------------------------------===//

// We use weak linking to provide a default implementation here. The XLA:GPU
// backend overrides this implementation, and it is picked at link time.
ABSL_ATTRIBUTE_WEAK XLA_FFI_Stream* GetXlaFfiStream(
    const CustomCall::UserData* user_data, const DiagnosticEngine* diagnostic) {
  return nullptr;
}

class FfiCustomCall : public CustomCall {
 public:
  FfiCustomCall(const XLA_FFI_Api* api, int64_t module_id,
                std::string_view name, XLA_FFI_Function* function)
      : api_(api), module_id_(module_id), name_(name), function_(function) {}

  std::string_view name() const final { return name_; }

  LogicalResult call(void** args, void** attrs, void** rets,
                     const UserData* user_data,
                     const DiagnosticEngine* diagnostic) const final {
    // Find an FFI module state for a given FFI call.
    FfiStateVector* state_vector =
        user_data ? user_data->getIfExists<FfiStateVector>() : nullptr;
    if (!state_vector || module_id_ >= state_vector->state.size())
      return diagnostic->EmitError(
          absl::InvalidArgumentError("FFI module state was not found"));

    // Prepare FFI execution context.
    XLA_FFI_ExecutionContext ctx;
    ctx.stream = GetXlaFfiStream(user_data, diagnostic);
    ctx.state = state_vector->state[module_id_];

    // Package custom call arguments and state into FFI function arguments.
    XLA_FFI_Function_Args ffi_args;
    ffi_args.struct_size = XLA_FFI_Function_Args_STRUCT_SIZE;
    ffi_args.priv = nullptr;
    ffi_args.api = api_;
    ffi_args.ctx = &ctx;
    ffi_args.args = args;
    ffi_args.attrs = attrs;
    ffi_args.rets = rets;

    // Execute FFI function and maybe report an error.
    if (XLA_FFI_Error* error = function_(&ffi_args)) {
      return diagnostic->EmitError(
          absl::Status(ConvertErrorCode(error->errc), error->error));
    }

    return success();
  }

 private:
  const XLA_FFI_Api* api_;
  int64_t module_id_;
  std::string name_;
  XLA_FFI_Function* function_;
};

//===----------------------------------------------------------------------===//
// FFI modules registered with the runtime.
//===----------------------------------------------------------------------===//

namespace {

class FfiModule;

// FfiState owns the opaque state created by the external FFI module, and
// destroys it using XLA FFI module API.
struct FfiState : public runtime::Module::State {
  FfiState(const FfiModule* parent, XLA_FFI_Module_State* state);
  ~FfiState() final;

  const FfiModule* parent;
  XLA_FFI_Module_State* state;
};

// Adaptor from the XLA FFI module and corresponding module API functions to the
// Xla runtime stateful module.
class FfiModule : public runtime::StatefulModule<FfiState> {
  using Base = runtime::StatefulModule<FfiState>;

 public:
  struct ExportedFunction {
    std::string_view name;
    XLA_FFI_Function* function;
  };

  FfiModule(const XLA_FFI_Api* api, int64_t module_id, const char* name,
            XLA_FFI_Module* module, XLA_FFI_Module_CreateState* create_state,
            XLA_FFI_Module_DestroyState* destroy_state,
            std::vector<ExportedFunction> exported_functions)
      : Base(name),
        api_(api),
        module_id_(module_id),
        module_(module),
        create_state_(create_state),
        destroy_state_(destroy_state),
        exported_functions_(std::move(exported_functions)) {}

  int64_t module_id() const { return module_id_; }

  absl::StatusOr<std::unique_ptr<FfiState>> CreateModuleState() const final;
  void DestroyState(XLA_FFI_Module_State* state) const;

  void Export(DynamicCustomCallRegistry& registry) const final;

 private:
  const XLA_FFI_Api* api_;
  int64_t module_id_;
  XLA_FFI_Module* module_;
  XLA_FFI_Module_CreateState* create_state_;
  XLA_FFI_Module_DestroyState* destroy_state_;
  std::vector<ExportedFunction> exported_functions_;
};

}  // namespace

FfiState::FfiState(const FfiModule* parent, XLA_FFI_Module_State* state)
    : parent(parent), state(state) {}

FfiState::~FfiState() { parent->DestroyState(state); }

absl::StatusOr<std::unique_ptr<FfiState>> FfiModule::CreateModuleState() const {
  if (!create_state_) return nullptr;

  XLA_FFI_Module_CreateState_Args args;
  args.struct_size = XLA_FFI_Module_CreateState_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.module = module_;
  args.state = nullptr;

  XLA_FFI_Error* error = create_state_(&args);
  if (error) return absl::InternalError(error->error);

  return std::make_unique<FfiState>(this, args.state);
}

void FfiModule::Export(DynamicCustomCallRegistry& registry) const {
  for (auto& fn : exported_functions_) {
    VLOG(1) << "Export FFI function: " << fn.name
            << " for module id: " << module_id_;
    registry.Register(std::make_unique<FfiCustomCall>(api_, module_id_, fn.name,
                                                      fn.function));
  }
}

void FfiModule::DestroyState(XLA_FFI_Module_State* state) const {
  if (!destroy_state_) return;

  XLA_FFI_Module_DestroyState_Args args;
  args.struct_size = XLA_FFI_Module_DestroyState_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.module = module_;
  args.state = state;

  destroy_state_(&args);
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
// XLA runtime FFI backend implementation.
//===----------------------------------------------------------------------===//

static std::vector<FfiModule>& OwnedFfiModules() {
  static auto* modules = new std::vector<FfiModule>();
  return *modules;
}

std::vector<const runtime::Module*> FfiModules() {
  std::vector<const runtime::Module*> modules;
  absl::c_transform(OwnedFfiModules(), std::back_inserter(modules),
                    [](const FfiModule& module) { return &module; });
  return modules;
}

void ExportFfiModules(DynamicCustomCallRegistry& registry) {
  for (auto* module : FfiModules()) module->Export(registry);
}

/*static*/ absl::StatusOr<FfiModulesState> FfiModulesState::Instantiate() {
  std::vector<std::unique_ptr<Module::State>> state;

  for (auto* module : FfiModules()) {
    auto module_state = module->CreateState();
    if (!module_state.ok()) return module_state.status();
    state.push_back(std::move(*module_state));
  }

  return FfiModulesState(std::move(state));
}

FfiModulesState::FfiModulesState(
    std::vector<std::unique_ptr<Module::State>> state)
    : state_(std::move(state)) {}

FfiStateVector FfiModulesState::state_vector() const {
  FfiStateVector state_vector;
  for (auto& state : state_) {
    auto* ffi_state = dynamic_cast<FfiState*>(state.get());
    state_vector.state.push_back(ffi_state ? ffi_state->state : nullptr);
  }
  return state_vector;
}

//===----------------------------------------------------------------------===//
// Implement XLA FFI module and function registration API.
//===----------------------------------------------------------------------===//

template <const XLA_FFI_Api* (*api)()>
static void RegisterXlaFfiModule(XLA_FFI_Module_Register_Args* args) {
  absl::Status struct_size_check = CheckMatchingStructSizes(
      "XLA_FFI_Module_Register_Args", XLA_FFI_Module_Register_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) LOG(ERROR) << struct_size_check.message();

  VLOG(1) << "Register FFI module: " << args->name;

  std::vector<FfiModule::ExportedFunction> exported_functions;
  for (int64_t i = 0; i < args->num_exported_functions; ++i) {
    FfiModule::ExportedFunction fn = {args->exported_names[i],
                                      args->exported_functions[i]};
    exported_functions.push_back(fn);
  }

  auto& modules = OwnedFfiModules();
  modules.emplace_back(api(), /*id=*/modules.size(), args->name, args->module,
                       args->create_state, args->destroy_state,
                       std::move(exported_functions));
}

static XLA_FFI_Module_State* GetXlaFfiModuleState(
    XLA_FFI_ExecutionContext_GetModuleState_Args* args) {
  absl::Status struct_size_check = CheckMatchingStructSizes(
      "XLA_FFI_ExecutionContext_GetModuleState_Args",
      XLA_FFI_ExecutionContext_GetModuleState_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) LOG(ERROR) << struct_size_check.message();

  return args->ctx->state;
}

static XLA_FFI_Stream* GetXlaFfiStream(
    XLA_FFI_ExecutionContext_GetStream_Args* args) {
  absl::Status struct_size_check = CheckMatchingStructSizes(
      "XLA_FFI_ExecutionContext_GetStream_Args",
      XLA_FFI_ExecutionContext_GetStream_Args_STRUCT_SIZE, args->struct_size);
  if (!struct_size_check.ok()) LOG(ERROR) << struct_size_check.message();

  return args->ctx->stream;
}

}  // namespace ffi
}  // namespace runtime
}  // namespace xla

template <typename T>
static XLA_FFI_TypeId FfiTypeId() {
  return xla::runtime::TypeID::get<xla::runtime::Tagged<T>>()
      .getAsOpaquePointer();
}

const XLA_FFI_Api ffi_api = {
    /*struct_size=*/XLA_FFI_Api_STRUCT_SIZE,
    /*priv=*/nullptr,

    //===------------------------------------------------------------------===//
    // Module Registration APIs.
    //===------------------------------------------------------------------===//
    ::xla::runtime::ffi::RegisterXlaFfiModule<GetXlaFfiApi>,

    //===------------------------------------------------------------------===//
    // Execution Context APIs.
    //===------------------------------------------------------------------===//
    ::xla::runtime::ffi::GetXlaFfiModuleState,
    ::xla::runtime::ffi::GetXlaFfiStream,

    //===------------------------------------------------------------------===//
    // Error Reporting APIs.
    //===------------------------------------------------------------------===//
    ::xla::runtime::ffi::CreateError,

    //===------------------------------------------------------------------===//
    // Type table.
    //===------------------------------------------------------------------===//
    FfiTypeId<float>,
    FfiTypeId<int32_t>,
    FfiTypeId<::xla::runtime::MemrefView>,
    FfiTypeId<::xla::runtime::StridedMemrefView>,
};

const XLA_FFI_Api* GetXlaFfiApi() { return &ffi_api; }
