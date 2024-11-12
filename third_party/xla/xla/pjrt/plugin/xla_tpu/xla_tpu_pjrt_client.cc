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

#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"

#include <memory>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_tpu.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "tsl/platform/statusor.h"

namespace xla {

const char kTpuPjrtName[] = "tpu";

absl::StatusOr<std::unique_ptr<PjRtClient>> GetXlaPjrtTpuClient() {
  const PJRT_Api* tpu_c_api = GetPjrtApi();
  if (!tpu_c_api) {
    return absl::InternalError("Failed to get PjrtApi");
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> tpu_client,
                      xla::WrapClientAroundCApi(tpu_c_api));

  if (tpu_client->platform_name() != kTpuPjrtName) {
    return absl::InternalError(
        absl::StrCat("Expected TPU client, got ", tpu_client->platform_name()));
  }

  return tpu_client;
}

}  // namespace xla
