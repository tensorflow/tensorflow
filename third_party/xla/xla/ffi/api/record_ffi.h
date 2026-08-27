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

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/record_api.h"

namespace xla::ffi {

struct RecordContext
    : public internal::RecordContextBase<internal::ErrorPolicy> {
  using Base = internal::RecordContextBase<internal::ErrorPolicy>;
  using Base::Base;
};

struct RecordExtension : public internal::RecordExtensionBase<RecordContext> {};

}  // namespace xla::ffi

#endif  // XLA_FFI_API_RECORD_FFI_H_
