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

#ifndef XLA_FFI_API_RECORD_FFI_H_
#define XLA_FFI_API_RECORD_FFI_H_

#ifdef XLA_FFI_RECORD_FFI_H_
#error Two different XLA FFI implementations cannot be included together. \
       See README.md for more details.
#endif  // XLA_FFI_API_RECORD_FFI_H_

#include <string>
#include <string_view>
#include <utility>

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/record_api.h"

namespace xla::ffi {

namespace internal {

inline Error ConvertError(XLA_FFI_Error* err) {
  const XLA_FFI_Api* api = XLA_FFI_GetApi();
  std::string msg = internal::GetErrorMessage(api, err);
  internal::DestroyError(api, err);
  return Error(ErrorCode::kInternal, std::move(msg));
}

// Error converter for external XLA:FFI.
struct ErrorConverter {
  template <typename T>
  static xla::ffi::ErrorOr<T> ToStatusOr(T value, XLA_FFI_Error* err) {
    if (err) {
      return xla::ffi::Unexpected(ConvertError(err));
    }
    return value;
  }

  static xla::ffi::Error ToStatus(XLA_FFI_Error* err) {
    if (err) {
      return ConvertError(err);
    }
    return xla::ffi::Error::Success();  // PASS-THROUGH
  }

  static xla::ffi::Unexpected<xla::ffi::Error> ToError(XLA_FFI_Error_Code err_c,
                                                       std::string_view msg) {
    return xla::ffi::Unexpected(xla::ffi::Error(err_c, std::string(msg)));
  }

  static xla::ffi::Error Success() { return xla::ffi::Error::Success(); }
};
}  // namespace internal

struct RecordContext
    : public internal::RecordContextBase<internal::ErrorConverter> {
  using Base = internal::RecordContextBase<internal::ErrorConverter>;
  using Base::Base;
};

struct RecordExtension : public internal::RecordExtensionBase<RecordContext> {};

}  // namespace xla::ffi

#endif  // XLA_FFI_API_RECORD_FFI_H_
