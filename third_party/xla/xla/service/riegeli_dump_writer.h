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

#ifndef XLA_SERVICE_RIEGELI_DUMP_WRITER_H_
#define XLA_SERVICE_RIEGELI_DUMP_WRITER_H_

#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/writer.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Creates a `riegeli::Writer` that writes data to a dump file. The file path is
// determined by the `debug_options`, `filename`, and by the `module` if its
// set.
//
// Similar to the other functions dump functions in dump.h, but can be used to
// avoid intermediate copies in some cases.
absl::StatusOr<std::unique_ptr<riegeli::Writer>> CreateRiegeliDumpWriter(
    const DebugOptions& debug_options, absl::string_view filename,
    const HloModule* absl_nullable module = nullptr);

}  // namespace xla

#endif  // XLA_SERVICE_RIEGELI_DUMP_WRITER_H_
