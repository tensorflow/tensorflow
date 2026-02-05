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

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/bytes/writer.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/dump.h"
#include "xla/service/dump_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla {

absl::StatusOr<std::unique_ptr<riegeli::Writer>> CreateRiegeliDumpWriter(
    const DebugOptions& debug_options, absl::string_view filename,
    const HloModule* module) {
  DumpOptions opts(debug_options);
  if (opts.dump_to.empty()) {
    return absl::InvalidArgumentError(
        "Dumping is not enabled (xla_dump_to is empty)");
  }
  if (opts.dumping_to_stdout()) {
    return absl::InvalidArgumentError(
        "Dumping to stdout is not supported for Riegeli writers");
  }

  std::string partial_path =
      module == nullptr ? std::string(filename)
                        : FilenameFor(*module, TimestampFor(*module), filename);

  TF_RETURN_IF_ERROR(CreateDirIfNeeded(opts.dump_to, tsl::Env::Default()));

  std::string file_path =
      tsl::io::JoinPath(opts.dump_to, SanitizeFileName(partial_path));
  return std::make_unique<riegeli::FdWriter<>>(file_path);
}

}  // namespace xla
