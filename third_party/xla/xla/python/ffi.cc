/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ffi.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/callback.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/py_host_callback.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace nb = nanobind;

class PyContext {
 public:
  enum Stage {
    kInstantiate = XLA_FFI_ExecutionStage_INSTANTIATE,
    kPrepare = XLA_FFI_ExecutionStage_PREPARE,
    kInitialize = XLA_FFI_ExecutionStage_INITIALIZE,
    kExecute = XLA_FFI_ExecutionStage_EXECUTE,
  };

  PyContext(const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx,
            XLA_FFI_ExecutionStage stage)
      : api_(api), ctx_(ctx), stage_(stage) {}

  Stage stage() const { return static_cast<Stage>(stage_); }
  absl::StatusOr<void*> stream() const {
    XLA_FFI_Stream_Get_Args args;
    args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = ctx_;
    args.stream = nullptr;
    if (XLA_FFI_Error* error = api_->XLA_FFI_Stream_Get(&args)) {
      return ffi::TakeStatus(error);
    }
    return args.stream;
  }

 private:
  const XLA_FFI_Api* api_;
  XLA_FFI_ExecutionContext* ctx_;
  XLA_FFI_ExecutionStage stage_;
};

class PyBuffer {
 public:
  explicit PyBuffer(const XLA_FFI_Buffer* buf) : buf_(buf) {}
  void* data() const { return buf_->data; }

 private:
  const XLA_FFI_Buffer* buf_;
};

template <XLA_FFI_ExecutionStage stage>
absl::Status FfiCallbackImpl(
    const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx,
    std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>* callbacks,
    uint64_t index, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  if (index >= callbacks->size()) {
    return absl::InvalidArgumentError("Callback index out of range.");
  }
  auto loaded_callback = llvm::dyn_cast_or_null<PyCpuLoadedHostCallback>(
      callbacks->at(index).get());
  if (loaded_callback == nullptr) {
    return absl::InternalError(
        "Expected a PyCpuLoadedHostCallback, got something else.");
  }
  CpuCallback* callback = loaded_callback->cpu_callback();

  nb::gil_scoped_acquire gil;
  auto nb_args =
      nb::steal<nb::tuple>(PyTuple_New(1 + args.size() + rets.size()));

  PyContext py_ctx(api, ctx, stage);
  PyTuple_SET_ITEM(nb_args.ptr(), 0, nb::cast(py_ctx).release().ptr());

  size_t offset = 1;
  for (size_t i = 0; i < args.size(); ++i, ++offset) {
    TF_ASSIGN_OR_RETURN(auto arg, args.get<ffi::AnyBuffer>(i));
    PyBuffer py_buffer(arg.buf());
    PyTuple_SET_ITEM(nb_args.ptr(), offset,
                     nb::cast(py_buffer).release().ptr());
  }

  for (size_t i = 0; i < rets.size(); ++i, ++offset) {
    TF_ASSIGN_OR_RETURN(auto ret, rets.get<ffi::AnyBuffer>(i));
    PyBuffer py_buffer(ret->buf());
    PyTuple_SET_ITEM(nb_args.ptr(), offset,
                     nb::cast(py_buffer).release().ptr());
  }

  EnterHostCallback();
  absl::StatusOr<nb::tuple> maybe_result_tuple = callback->FfiCall(nb_args);
  LeaveHostCallback();
  TF_RETURN_IF_ERROR(maybe_result_tuple.status());

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(
    kFfiCallback, FfiCallbackImpl<XLA_FFI_ExecutionStage_EXECUTE>,
    ffi::Ffi::Bind()
        .Ctx<ffi::FfiApi>()
        .Ctx<ffi::FfiExecutionContext>()
        .Ctx<ffi::UserData<
            std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>>>()
        .Attr<uint64_t>("index")
        .RemainingArgs()
        .RemainingRets());
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_python_buffer_callback",
                         "Host", kFfiCallback);

void BuildFfiSubmodule(nb::module_& m) {
  nb::module_ ffi_module =
      m.def_submodule("ffi", "Python bindings for the XLA FFI.");

  nb::class_<PyBuffer> buffer(ffi_module, "Buffer");
  buffer.def("data", &PyBuffer::data);

  nb::enum_<PyContext::Stage>(ffi_module, "ExecutionStage")
      .value("INSTANTIATE", PyContext::Stage::kInstantiate)
      .value("PREPARE", PyContext::Stage::kPrepare)
      .value("INITIALIZE", PyContext::Stage::kInitialize)
      .value("EXECUTE", PyContext::Stage::kExecute)
      .export_values();

  nb::class_<PyContext> context(ffi_module, "ExecutionContext");
  context.def("stage", &PyContext::stage);
  context.def("stream", ValueOrThrowWrapper(&PyContext::stream));
}

}  // namespace xla
