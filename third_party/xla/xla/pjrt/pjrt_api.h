/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_API_H_
#define XLA_PJRT_PJRT_API_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "tsl/platform/platform.h"

namespace pjrt {

// Gets and sets the global map for PJRT_Api*. Not thread safe. `device_type` is
// case insensitive.
absl::StatusOr<const PJRT_Api*> PjrtApi(absl::string_view device_type);
absl::Status SetPjrtApi(absl::string_view device_type, const PJRT_Api* api);

// Loads a PJRT plugin. The library provided by library_path must export a
// symbol called `GetPjrtApi` with function signature `const PJRT_Api*
// GetPjrtApi()`. This method dlopen the plugin library, dlsym `GetPjrtApi`,
// calls `GetPjrtApi` and `SetPjrtApi`. Returns the loaded PJRT_Api* if
// successful.
absl::StatusOr<const PJRT_Api*> LoadPjrtPlugin(absl::string_view device_type,
                                               absl::string_view library_path);

// Requires that SetPjrtApi has been successfully called on `device_type` before
// calling this method.
absl::StatusOr<bool> IsPjrtPluginInitialized(absl::string_view device_type);
// Initializes a PJRT plugin with `PJRT_Plugin_Initialize`.
absl::Status InitializePjrtPlugin(absl::string_view device_type);

}  // namespace pjrt

#endif  // XLA_PJRT_PJRT_API_H_
