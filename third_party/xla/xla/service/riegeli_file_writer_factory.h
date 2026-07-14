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

#ifndef XLA_SERVICE_RIEGELI_FILE_WRITER_FACTORY_H_
#define XLA_SERVICE_RIEGELI_FILE_WRITER_FACTORY_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "riegeli/bytes/writer.h"

namespace xla {

// Creates a `riegeli::Writer` that will write to the given filename.
//
// If the file doesn't exist it will be created, and if it does, it will be
// truncated.
//
// This factory is needed since we need a different implementation for Google
// internal use that can handle Google's infra (which isn't available in OSS).
std::unique_ptr<riegeli::Writer> CreateRiegeliFileWriter(
    absl::string_view filename);

}  // namespace xla

#endif  // XLA_SERVICE_RIEGELI_FILE_WRITER_FACTORY_H_
