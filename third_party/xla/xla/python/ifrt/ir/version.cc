/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/version.h"

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

namespace xla {
namespace ifrt {

namespace {

static int64_t parseNumber(llvm::StringRef num_ref) {
  int64_t num;
  if (num_ref.getAsInteger(/*radix=*/10, num)) {
    llvm::report_fatal_error("failed to parse version number");
  }
  return num;
}

/// Validate version argument is `#.#.#` (ex: 0.9.0, 0.99.0, 1.2.3)
/// Returns the vector of 3 matches (major, minor, patch) if successful,
/// else returns failure.
static mlir::FailureOr<llvm::SmallVector<int64_t, 3>> extractVersionNumbers(
    llvm::StringRef version_ref) {
  llvm::Regex versionRegex("^([0-9]+)\\.([0-9]+)\\.([0-9]+)$");
  llvm::SmallVector<llvm::StringRef> matches;
  if (!versionRegex.match(version_ref, &matches)) {
    return mlir::failure();
  }
  return llvm::SmallVector<int64_t, 3>{parseNumber(matches[1]),
                                       parseNumber(matches[2]),
                                       parseNumber(matches[3])};
}

}  // namespace

mlir::FailureOr<Version> Version::fromString(llvm::StringRef version_ref) {
  auto failOrVersionArray = extractVersionNumbers(version_ref);
  if (mlir::failed(failOrVersionArray)) {
    return mlir::failure();
  }
  auto versionArr = *failOrVersionArray;
  return Version(versionArr[0], versionArr[1], versionArr[2]);
}

mlir::FailureOr<int64_t> Version::getBytecodeVersion() const {
  if (*this <= getCurrentVersion()) return 0;
  return mlir::failure();
}

Version Version::fromCompatibilityRequirement(
    CompatibilityRequirement requirement) {
  switch (requirement) {
    case CompatibilityRequirement::NONE:
      return Version::getCurrentVersion();
    case CompatibilityRequirement::WEEK_4:
      return Version(0, 1, 0);  // v0.1.0 - Nov 05, 2024
    case CompatibilityRequirement::WEEK_12:
      return Version(0, 1, 0);  // v0.1.0 - Nov 05, 2024
    case CompatibilityRequirement::MAX:
      return Version::getMinimumVersion();
  }
  llvm::report_fatal_error("Unsupported compatibility requirement");
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const Version& version) {
  return diag << version.toString();
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Version& version) {
  return os << version.toString();
}

}  // namespace ifrt
}  // namespace xla
