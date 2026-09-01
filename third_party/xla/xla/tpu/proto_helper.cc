/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/tpu/proto_helper.h"

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "xla/tpu/c_api_decl.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

extern "C" {

void StreamExecutor_Tpu_FreeSerializedProto(
    const TpuSerializedProto* absl_nonnull proto) {
  CHECK_NE(proto, nullptr);
  // delete[] nullptr is a safe no-op in C++.
  delete[] proto->bytes;
}

}  // extern "C"
