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

#ifndef XLA_PYTHON_IFRT_IR_VERSION_H_
#define XLA_PYTHON_IFRT_IR_VERSION_H_

#include <cstdint>
#include <sstream>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace xla {
namespace ifrt {

class Version {
 public:
  // Convenience method to extract major, minor, patch and create a Version
  // from a StringRef of the form `#.#.#`. Returns failure if the string is
  // invalid.
  static mlir::FailureOr<Version> fromString(llvm::StringRef version_ref);

  // Returns a Version representing the current IFRT IR version.
  static Version getCurrentVersion() { return Version(0, 1, 0); }

  /// Returns a Version representing the minimum supported IFRT IR version.
  static Version getMinimumVersion() { return Version(0, 1, 0); }

  // CompatibilityRequirement is used to get a viable target version to use for
  // `xla::ifrt::Serialize` given a compatibility requirement specified as a
  // duration.
  //
  // Values represent a minimum requirement, i.e. WEEK_4 will return a version
  // that is at least 4 weeks old.
  enum class CompatibilityRequirement {
    NONE = 0,     // No compat requirement, use latest version.
    WEEK_4 = 1,   // 1 month requirement
    WEEK_12 = 2,  // 3 month requirement
    MAX = 3,      // Maximum compat, use minimum supported version
  };

  // Get a viable target version to use for `xla::ifrt::Serialize` for a given
  // compatibility requirement.
  static Version fromCompatibilityRequirement(
      CompatibilityRequirement requirement);

  // Return the MLIR Bytecode version associated with the IFRT IR version
  // instance. Returns failure if the version is not in compatibility window.
  mlir::FailureOr<int64_t> getBytecodeVersion() const;

  // Construct Version from major, minor, patch integers.
  Version(int64_t major, int64_t minor, int64_t patch)
      : major_minor_patch_({major, minor, patch}) {}

  int64_t getMajor() const { return major_minor_patch_[0]; }
  int64_t getMinor() const { return major_minor_patch_[1]; }
  int64_t getPatch() const { return major_minor_patch_[2]; }

  bool operator<(const Version& other) const {
    return major_minor_patch_ < other.major_minor_patch_;
  }
  bool operator==(const Version& other) const {
    return major_minor_patch_ == other.major_minor_patch_;
  }
  bool operator<=(const Version& other) const {
    return major_minor_patch_ <= other.major_minor_patch_;
  }
  std::string toString() const {
    std::ostringstream os;
    os << getMajor() << '.' << getMinor() << '.' << getPatch();
    return os.str();
  }

 private:
  llvm::SmallVector<int64_t, 3> major_minor_patch_;
};

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const Version& version);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Version& version);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_VERSION_H_
