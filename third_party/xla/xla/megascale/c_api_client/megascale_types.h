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

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/megascale/addresses.pb.h"
#include "xla/megascale/dcn_topology.pb.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_megascale_extension.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace megascale {
namespace c_api_client {

class MultiSliceDeviceId {
 public:
  static absl::StatusOr<MultiSliceDeviceId> Create(int64_t megascale_id);
  static absl::StatusOr<MultiSliceDeviceId> Create(int32_t slice_id,
                                                   int32_t per_slice_device_id);

  MultiSliceDeviceId(const MultiSliceDeviceId&) = default;
  MultiSliceDeviceId& operator=(const MultiSliceDeviceId&) = default;

  int32_t per_slice_device_id() const { return per_slice_device_id_; }
  int32_t slice_id() const { return slice_id_; }
  int64_t megascale_id() const { return megascale_id_; }

  bool operator==(const MultiSliceDeviceId& rhs) const {
    return megascale_id_ == rhs.megascale_id_;
  }
  bool operator!=(const MultiSliceDeviceId& rhs) const {
    return megascale_id_ != rhs.megascale_id_;
  }
  template <typename H>
  friend H AbslHashValue(H h, const MultiSliceDeviceId& m) {
    return H::combine(std::move(h), m.megascale_id_);
  }

 private:
  explicit MultiSliceDeviceId(int64_t megascale_id, int32_t slice_id,
                              int32_t per_slice_device_id)
      : megascale_id_(megascale_id),
        slice_id_(slice_id),
        per_slice_device_id_(per_slice_device_id) {}

  int64_t megascale_id_;
  int32_t slice_id_;
  int32_t per_slice_device_id_;
};

class CApiPjRtClientContext {
 public:
  CApiPjRtClientContext(PJRT_Megascale_ClientContext* client_context,
                        const PJRT_Api* c_api,
                        const PJRT_Megascale_Extension* extension)
      : client_context_(client_context),
        c_api_(c_api),
        extension_(CHECK_NOTNULL(extension)) {}

  ~CApiPjRtClientContext();

  PJRT_Megascale_ClientContext* get() const { return client_context_; }

  absl::Status Initialize();

  absl::Status UnblockPendingWork(int32_t launch_id,
                                  absl::Duration expire_after);

  absl::StatusOr<int> megascale_port();

  const PJRT_Api* c_api() const { return c_api_; }
  const PJRT_Megascale_Extension* extension() const { return extension_; }

 private:
  PJRT_Megascale_ClientContext* client_context_;
  const PJRT_Api* c_api_;
  const PJRT_Megascale_Extension* extension_;
};

}  // namespace c_api_client
}  // namespace megascale
}  // namespace xla

#endif  // XLA_MEGASCALE_C_API_CLIENT_MEGASCALE_TYPES_H_
