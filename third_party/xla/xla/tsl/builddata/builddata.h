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

#ifndef XLA_TSL_BUILDDATA_BUILDDATA_H_
#define XLA_TSL_BUILDDATA_BUILDDATA_H_

#include <cstdint>
#include <ctime>
#include <string>

#include "absl/strings/string_view.h"

namespace tsl::builddata {

// The status of the version control workspace in which the binary was built.
enum ClientStatusType {
  MINT,      // The client was mint (unmodified) (e.g., `git status` showed a
             // clean working tree).
  MODIFIED,  // The workspace contained local, uncommitted modifications.
  UNKNOWN,   // The workspace status could not be determined.
};

// Information about who built this binary and where it was built.
// This typically combines the username, hostname, and the directory path.
//
// An example would be:
//     "user@build-machine:/path/to/workspace"
absl::string_view BuildInfo();

// A unique identifier for the build execution, if set by the build system.
// This is typically an opaque identifier (such as a UUID) that uniquely
// identifies the specific build invocation. It can be used to correlate
// the binary with external build logging or result storage systems.
//
// Example:
//     "e070867e-9496-4e71-8439-5c4374ee33df"
absl::string_view BuildId();

// Where (the client root) this binary was built.
// Just a standard accessor to the piece in BuildInfo().
//
// An example would be:
//     "/home/jeff/xla/"
absl::string_view BuildDir();

// A URI representing the source code location or state from which this
// binary was built. This can be used to trace the binary back to the exact
// source tree or snapshot.
//
// For now it simply returns the same as SourceRevision().
absl::string_view SourceUri();

// Where (the host) this binary was built.
// Just a standard accessor to the piece in BuildInfo().
//
// An example would be:
//     "build-machine-01.example.com"
absl::string_view BuildHost();

// The file path relative to BuildDir() for where this binary
// was originally built at (used for self-identification).
// No guarantees that this same binary version is still at this location.
//
// An example would be:
//     "bazel-out/k8-fastbuild/bin/my_test"
absl::string_view BuildTarget();

// This is the build label (target name) for this binary (used for
// self-identification). This is a build argument for bazel, contrast it with
// BuildTarget() above, which is the exact file path.
//
// An example would be:
//     "@@//:my_test"
absl::string_view TargetName();

// A custom build label embedded into the binary.
// This returns the value passed to the build system during
// compilation via the `--embed_label` flag in Bazel. Returns
// an empty string if no label was provided.
//
// An example would be:
//     "v1.2.3-release"
absl::string_view BuildLabel();

// The workspace in which the build was performed; the precise format is not
// specified and not to be depended on.
absl::string_view BuildClient();

// When this binary was built.
//
// An example would be:
//     "Built on Jan  5 2005 14:24:56 [TZ=America/Los_Angeles] (1104963895)"
//
// Note: This timestamp is formatted in the America/Los_Angeles timezone
// (PDT/PST) to maintain consistency with existing users and avoid
// discrepancies in logs between internal and open-source environments.
std::string Timestamp();

// When this binary was built (used for versioning).
// It encodes the time since epoch in seconds.
// An example would be:
//     1104963895
time_t TimestampAsInt();

// The source revision (changelist number or git hash) from the source code
// repository that was used to build this binary.
//
// Returns:
//    - Empty string if Undefined("").
//    - "<unknown>" if it could not be determined.
//    - The changelist number or git hash otherwise.
absl::string_view SourceRevision();

// Positive integer value of SourceRevision()
// Returns:
//    > 0 if SCM revision is a positive integer
//    0 if Unknown("<unknown>") (could not be determined)
//   -1 if Undefined("")
//   -2 if SCM revision is not a valid integer (like a git hash)
int64_t SourceRevisionAsInt();

// If the binary was built in a branch, returns the source revision of
// mainline that was used as the base for that branch.  Otherwise
// returns the same value as SourceRevision().
//
// Returns:
//    - Empty string if Undefined("").
//    - "<unknown>" if it could not be determined.
//    - The SCM revision or git hash otherwise.
absl::string_view BaselineSourceRevision();

// Positive integer value of BaselineSourceRevision()
// Returns:
//    > 0 if SCM revision is a positive integer
//    0 if Unknown("<unknown>") (could not be determined)
//   -1 if Undefined("")
//   -2 if SCM revision is not a valid integer (like a git hash)
int64_t BaselineSourceRevisionAsInt();

// The status of the version control workspace in which the binary was built.
// -DBUILD_CLIENT_MINT_STATUS supplies the information for this function.
ClientStatusType ClientStatus();

// String value of ClientStatus(): one of "mint", "modified", "unknown"
// (used as an exported variable).
absl::string_view ClientStatusAsString();

// The compiler/target-cpu combination used to compile this executable.
//
// An example would be:
//     "gcc-4.6.x-glibc-2.11.1-grte-cxx11-k8"
absl::string_view CompilerTarget();

}  // namespace tsl::builddata

#endif  // XLA_TSL_BUILDDATA_BUILDDATA_H_
