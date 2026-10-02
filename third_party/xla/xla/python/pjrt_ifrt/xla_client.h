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

#ifndef XLA_PYTHON_PJRT_IFRT_XLA_CLIENT_H_
#define XLA_PYTHON_PJRT_IFRT_XLA_CLIENT_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// Wraps loading options for an XLA computation.
//
// TODO(hyeontaek): Move this class out of pjrt_ifrt.
struct XlaLoadOptions : RTTIExtends<XlaLoadOptions, LoadOptions> {
  XlaLoadOptions() = default;

  // NOLINTBEGIN(clang-diagnostic-shadow-field)
  XlaLoadOptions(
      DeviceListRef devices,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks,
      std::optional<std::vector<int>> outputs_bundle_slice_sizes)
      : RTTIExtends<XlaLoadOptions, LoadOptions>(
            std::move(devices), std::move(outputs_bundle_slice_sizes)),
        loaded_host_callbacks(std::move(loaded_host_callbacks)) {}
  // NOLINTEND(clang-diagnostic-shadow-field)

  std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks;

  // LoadOptions implementation.

  ~XlaLoadOptions() override = default;

  static char ID;  // NOLINT
};

// Gets `xla::ifrt::XlaLoadOptions` from `xla::ifrt::LoadOptions`.
absl::StatusOr<std::unique_ptr<XlaLoadOptions>> GetXlaLoadOptions(
    std::unique_ptr<LoadOptions> options);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_XLA_CLIENT_H_
