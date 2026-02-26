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

#include "xla/tsl/util/memfile_builtin.h"

#include <string>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/embedded_filesystem.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/util/file_toc.h"
#include "tsl/platform/path.h"

namespace tsl::memfile {

static constexpr absl::string_view kScheme = "embed";
static constexpr absl::string_view kSchemeUri = "embed://";

EmbedFileSystem& global_file_system() {
  static absl::NoDestructor<EmbedFileSystem> global_file_system;
  return *global_file_system;
}

absl::Status RegisterBuiltInFiles(const char* absl_nonnull name,
                                  const FileToc toc[absl_nonnull]) {
  for (; toc->name != nullptr; ++toc) {
    absl::string_view contents(toc->data, toc->size);
    const std::string path =
        tsl::io::JoinPath(kSchemeUri, name, tsl::io::Basename(toc->name));
    // It would be nice to log these, but we don't have a way to do it
    // conditionally. We're running at global-init time, before flags have
    // parsed, so VLOG is out, and any standard log level will result in RAW_LOG
    // on stderr.
    TF_RETURN_IF_ERROR(global_file_system().EmbedFile(path, contents));
  }

  return absl::OkStatus();
}

bool GlobalRegisterFiles(const char* absl_nonnull name,
                         const FileToc toc[absl_nonnull]) {
  tsl::FileSystem* sys;

  // Inline some of the definition of REGISTER_FILE_SYSTEM so we can fetch our
  // singleton back out. Multiple registrations is an error.
  if (!tsl::Env::Default()
           ->GetFileSystemForFile(tsl::io::JoinPath(kSchemeUri, "dummy"), &sys)
           .ok()) {
    QCHECK_OK(tsl::Env::Default()->RegisterFileSystem(
        std::string(kScheme),
        []() -> tsl::FileSystem* { return &global_file_system(); }));
  }

  QCHECK_OK(RegisterBuiltInFiles(name, toc));
  return true;
}

}  // namespace tsl::memfile
