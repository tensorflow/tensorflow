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
#include "absl/status/status.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/ffi/ffi.h"

namespace xla {

namespace {

using F32Scalar = ffi::BufferR0<PrimitiveType::F32>;

auto AlwaysFail(ffi::Result<ffi::BufferBase>) {
  return absl::InternalError("Failed intentionally");
}

XLA_FFI_DEFINE_HANDLER(kAlwaysFail, AlwaysFail,
                       ffi::Ffi::Bind().Ret<ffi::BufferBase>());

auto AlwaysSucceed(ffi::Result<ffi::BufferBase>) { return absl::OkStatus(); }

XLA_FFI_DEFINE_HANDLER(kAlwaysSucceed, AlwaysSucceed,
                       ffi::Ffi::Bind().Ret<ffi::BufferBase>());

auto SubtractN(F32Scalar in, ffi::Result<F32Scalar> out, float n) {
  *out->data.base() = *in.data.base() - n;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kSubtractN, SubtractN,
                       ffi::Ffi::Bind()
                           .Arg<F32Scalar>()  // in
                           .Ret<F32Scalar>()  // out
                           .Attr<float>("n"));

namespace nb = ::nanobind;

template <typename T>
auto BindFunction(T* fn) {
  return nb::capsule(absl::bit_cast<void*>(fn));
}

NB_MODULE(typed_ffi_custom_calls, m) {
  m.def("registrations", []() {
    nb::dict dict;
    dict["xla_client_test$$AlwaysFail"] = BindFunction(kAlwaysFail);
    dict["xla_client_test$$AlwaysSucceed"] = BindFunction(kAlwaysSucceed);
    dict["xla_client_test$$subtract_n_f32"] = BindFunction(kSubtractN);
    return dict;
  });
}

}  // namespace

}  // namespace xla
