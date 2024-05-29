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

#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace xla::ffi {
namespace nb = ::nanobind;

// Implement custom calls as static functions with XLA FFI types in the function
// signature that gives access to the arguments and results buffers together
// with their types and dimensions. See `ffi/api/ffi_test.cc` for more XLA FFI
// examples and features (e.g. binding attributes, custom user-defined structs
// and arbitrary execution context).

static Error AlwaysFail(Result<AnyBuffer>) {
  return Error(XLA_FFI_Error_Code_INTERNAL, "Failed intentionally");
}

static Error AlwaysSucceed(Result<AnyBuffer>) { return Error::Success(); }

static Error Subtract(BufferR0<DataType::F32> a, BufferR0<DataType::F32> b,
                      Result<BufferR0<DataType::F32>> out) {
  *out->data = *a.data - *b.data;
  return Error::Success();
}

static Error SubtractCst(BufferR0<DataType::F32> a,
                         Result<BufferR0<DataType::F32>> out, float cst) {
  *out->data = *a.data - cst;
  return Error::Success();
}

// Define XLA FFI handlers from the implementations defined above using explicit
// XLA FFI binding API to describe type signatures of custom calls.

XLA_FFI_DEFINE_HANDLER(kAlwaysFail, AlwaysFail, Ffi::Bind().Ret<AnyBuffer>());

XLA_FFI_DEFINE_HANDLER(kAlwaysSucceed, AlwaysSucceed,
                       Ffi::Bind().Ret<AnyBuffer>());

XLA_FFI_DEFINE_HANDLER(kSubtract, Subtract,
                       Ffi::Bind()
                           .Arg<BufferR0<DataType::F32>>()
                           .Arg<BufferR0<DataType::F32>>()
                           .Ret<BufferR0<DataType::F32>>());

XLA_FFI_DEFINE_HANDLER(kSubtractCst, SubtractCst,
                       Ffi::Bind()
                           .Arg<BufferR0<DataType::F32>>()
                           .Ret<BufferR0<DataType::F32>>()
                           .Attr<float>("cst"));

template <typename T>
static auto BindFunction(T* fn) {
  return nb::capsule(reinterpret_cast<void*>(fn));
}

// Custom calls registration library that exports function pointers to XLA FFI
// handlers to the python users.
NB_MODULE(custom_calls_testlib, m) {
  m.def("registrations", []() {
    nb::dict dict;
    dict["always_fail"] = BindFunction(kAlwaysFail);
    dict["always_succeed"] = BindFunction(kAlwaysSucceed);
    dict["subtract_f32"] = BindFunction(kSubtract);
    dict["subtract_f32_cst"] = BindFunction(kSubtractCst);
    return dict;
  });
}

}  // namespace xla::ffi
