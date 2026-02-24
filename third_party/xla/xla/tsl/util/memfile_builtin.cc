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

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/ram_file_system.h"
#include "xla/tsl/util/file_toc.h"
#include "tsl/platform/path.h"

namespace tsl::memfile {

absl::Status RegisterBuiltInFiles(const char* absl_nonnull name,
                                  const FileToc toc[absl_nonnull]) {
  // We register our own RamFileSystem to avoid any initialization-order
  // concerns with the ram:// filesystem.
  tsl::FileSystem* sys;
  std::string scheme("embed");
  std::string uri_scheme = tsl::io::CreateURI(scheme, /*host=*/"", "");
  if (!tsl::Env::Default()
           ->GetFileSystemForFile(tsl::io::JoinPath(uri_scheme, "dummy"), &sys)
           .ok()) {
    TF_RETURN_IF_ERROR(tsl::Env::Default()->RegisterFileSystem(
        scheme, std::make_unique<RamFileSystem>(scheme)));
  }

  for (; toc->name != nullptr; ++toc) {
    absl::string_view contents(toc->data, toc->size);
    const std::string path =
        tsl::io::JoinPath(uri_scheme, name, tsl::io::Basename(toc->name));
    VLOG(1) << "Registering memfile at " << path;
    TF_RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), path, contents));
  }

  return absl::OkStatus();
}

bool GlobalRegisterFiles(const char* absl_nonnull name,
                         const FileToc toc[absl_nonnull]) {
  QCHECK_OK(RegisterBuiltInFiles(name, toc));
  return true;
}

}  // namespace tsl::memfile
