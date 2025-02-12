/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "absl/status/status.h"
#include "xla/tsl/platform/default/dso_loader.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {
namespace internal {
namespace DsoLoader {

// Skip check when GPU libraries are statically linked.
absl::Status MaybeTryDlopenGPULibraries() {
  LOG(INFO) << "GPU libraries are statically linked, skip dlopen check.";
  return absl::OkStatus();
}
}  // namespace DsoLoader
}  // namespace internal
}  // namespace tsl
