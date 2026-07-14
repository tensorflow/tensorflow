/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt_proxy/common/transfer_util.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace ifrt {
namespace proxy {

constexpr char kLargeTransferOptimizationDirectoryEnv[] =
    "IFRT_PROXY_GRPC_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY";

// Returns the environmental variable kLargeTransferOptimizationDirectoryEnv
// with `$$TEST_TMPDIR$$` substituted by any temporary directory.
static std::string UncachedLargeTransferOptimizationDirectory() {
  const char* valptr = std::getenv(kLargeTransferOptimizationDirectoryEnv);
  if (valptr == nullptr) {
    return "";
  }

  std::string result = valptr;
  if (!absl::StrContains(result, "$$TEST_TMPDIR$$")) {
    return result;
  }

  std::vector<std::string> tmp_dirs;
  tsl::Env::Default()->GetLocalTempDirectories(&tmp_dirs);
  CHECK(!tmp_dirs.empty())
      << "Environmental variable " << kLargeTransferOptimizationDirectoryEnv
      << " contains $$TEST_TMPDIR$$ but unable to find a temporary directory.";

  return absl::StrReplaceAll(result, {{"$$TEST_TMPDIR$$", tmp_dirs[0]}});
}

static absl::string_view LargeTransferOptimizationDirectory() {
  static absl::NoDestructor<std::string> result(
      UncachedLargeTransferOptimizationDirectory());
  LOG_EVERY_N(INFO, 1) << kLargeTransferOptimizationDirectoryEnv << "="
                       << *result;
  return *result;
}

std::optional<std::string> LargeTransferFilePath(int handle) {
  if (LargeTransferOptimizationDirectory().empty()) {
    return std::nullopt;
  }
  return absl::StrCat(LargeTransferOptimizationDirectory(), "/lt_", handle);
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
