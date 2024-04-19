// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_RUNTIME_AOT_FFI_H_
#define XLA_RUNTIME_AOT_FFI_H_

#include "xla/runtime/ffi/ffi_api.h"

namespace xla {
namespace runtime {
namespace aot {

XLA_FFI_Api FfiApi();

XLA_FFI_Function_Args FfiArgs(XLA_FFI_Api* api, void** args, void** attrs,
                              void** rets);

bool ProcessErrorIfAny(XLA_FFI_Error* error);

}  // namespace aot
}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_AOT_FFI_H_
