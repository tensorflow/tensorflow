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

#ifndef XLA_MEGASCALE_C_API_CLIENT_MEGASCALE_TYPES_H_
#define XLA_MEGASCALE_C_API_CLIENT_MEGASCALE_TYPES_H_

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/megascale/addresses.pb.h"
#include "xla/megascale/dcn_topology.pb.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_megascale_extension.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace megascale {
namespace c_api_client {

class CApiPjRtClientContext {
 public:
  CApiPjRtClientContext(PJRT_Megascale_ClientContext* client_context,
                        const PJRT_Api* c_api,
                        PJRT_Megascale_Extension* extension)
      : client_context_(client_context),
        c_api_(c_api),
        extension_(CHECK_NOTNULL(extension)) {}

  ~CApiPjRtClientContext();

  PJRT_Megascale_ClientContext* get() const { return client_context_; }

  absl::Status Initialize();

  absl::Status UnblockPendingWork(int32_t launch_id,
                                  absl::Duration expire_after);

  absl::StatusOr<int> megascale_port();

 private:
  PJRT_Megascale_ClientContext* client_context_;
  const PJRT_Api* c_api_;
  const PJRT_Megascale_Extension* extension_;
};

class PjRtCApiMultiSliceConfig : public xla::MultiSliceConfig {
 public:
  PjRtCApiMultiSliceConfig(PJRT_MultiSlice_Config* config,
                           const PJRT_Api* c_api,
                           PJRT_Megascale_Extension* extension)
      : config_(config), c_api_(c_api), extension_(CHECK_NOTNULL(extension)) {}

  ~PjRtCApiMultiSliceConfig() override;

  int32_t NumSlices() const override;
  int32_t SliceId() const override;
  absl::flat_hash_map<int32_t, int32_t> NumDevicesPerSlice() const override;
  std::string Serialize() const override;
  PJRT_MultiSlice_Config* get() const { return config_; }

 private:
  PJRT_MultiSlice_Config* config_;
  const PJRT_Api* c_api_;
  const PJRT_Megascale_Extension* extension_;
};

}  // namespace c_api_client
}  // namespace megascale
}  // namespace xla

#endif  // XLA_MEGASCALE_C_API_CLIENT_MEGASCALE_TYPES_H_
