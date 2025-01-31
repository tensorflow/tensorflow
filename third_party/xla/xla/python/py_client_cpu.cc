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

#include "xla/python/py_client_cpu.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/python/callback.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/py_host_callback.h"
#include "xla/python/types.h"
#include "xla/service/custom_call_status.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/statusor.h"

namespace nb = nanobind;

namespace xla {

absl::Status FfiPrepareAndCall(CpuCallback* callback, ffi::RemainingRets rets,
                               ffi::RemainingArgs args) {
  nb::gil_scoped_acquire gil;
  nb::tuple nb_args = nb::steal<nb::tuple>(PyTuple_New(args.size()));

  for (size_t i = 0; i < args.size(); ++i) {
    auto arg = args.get<ffi::AnyBuffer>(i);
    auto dtype = arg->element_type();
    if (dtype == TOKEN) {
      PyTuple_SET_ITEM(nb_args.ptr(), i, nb::none().release().ptr());
    } else {
      auto dims = absl::Span<const int64_t>(arg->dimensions().begin(),
                                            arg->dimensions().size());
      TF_ASSIGN_OR_RETURN(auto nb_dtype, PrimitiveTypeToNbDtype(dtype));
      nb_numpy_ndarray array = nb_numpy_ndarray(
          nb_dtype, dims, std::nullopt,
          const_cast<void*>(
              args.get<ffi::AnyBuffer>(i).value().untyped_data()));
      array.attr("flags").attr("writeable") = nb::bool_(false);
      PyTuple_SET_ITEM(nb_args.ptr(), i, array.release().ptr());
    }
  }

  TF_ASSIGN_OR_RETURN(auto result_tuple,
                      callback->FfiCall(rets, std::move(nb_args)));

  for (size_t i = 0; i < rets.size(); ++i) {
    auto ret = rets.get<ffi::AnyBuffer>(i);
    auto dtype = ret.value()->element_type();
    if (dtype == TOKEN) {
      continue;
    }
    nb::object output =
        nb::borrow<nb::object>(PyTuple_GetItem(result_tuple.ptr(), i));
    nb_numpy_ndarray array = nb_numpy_ndarray::ensure(std::move(output));
    absl::Span<int64_t const> dims(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    // TODO(dsuo): What if we don't get the expected stride?
    std::memcpy(rets.get<ffi::AnyBuffer>(i).value()->untyped_data(),
                array.data(), ret.value()->size_bytes());
  }

  return absl::OkStatus();
}

void XlaPythonCpuCallback(void* output, void** inputs,
                          XlaCustomCallStatus* status) {
  CpuCallback* callback =
      absl::bit_cast<CpuCallback*>(*static_cast<uintptr_t*>(inputs[0]));
  auto s = callback->PrepareAndCall(output, inputs + 1);
  if (!s.ok()) {
    auto msg = s.message();
    XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
  }
}

absl::Status XlaFfiPythonCpuCallbackImpl(
    std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>* callbacks,
    uint64_t index, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  auto host_callback = callbacks->at(index).get();
  auto py_host_callback =
      llvm::dyn_cast_or_null<PyCpuLoadedHostCallback>(host_callback);

  if (py_host_callback == nullptr) {
    return absl::InternalError(
        "Expected a PyCpuLoadedHostCallback, got something else.");
  }
  // TODO(dsuo): We could skip this indirection if we could access the
  // CpuCallback directly.
  auto descriptor = py_host_callback->descriptor();
  CpuCallback* callback =
      absl::bit_cast<CpuCallback*>(static_cast<uintptr_t>(descriptor));
  return FfiPrepareAndCall(callback, rets, args);
}

XLA_FFI_DEFINE_HANDLER(
    XlaFfiCpuCallback, XlaFfiPythonCpuCallbackImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::UserData<
            std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>>>()
        .Attr<uint64_t>("index")
        .RemainingArgs()
        .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_ffi_python_cpu_callback",
                         "HOST", XlaFfiCpuCallback);

}  // namespace xla
