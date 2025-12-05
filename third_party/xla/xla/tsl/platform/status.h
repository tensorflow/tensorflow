/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PLATFORM_STATUS_H_
#define XLA_TSL_PLATFORM_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/stack_frame.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tsl/platform/platform.h"

// Include appropriate platform-dependent parts of status.
#if defined(PLATFORM_GOOGLE)
#include "xla/tsl/platform/google/status.h"  // IWYU pragma: export
#else
#include "xla/tsl/platform/default/status.h"  // IWYU pragma: export
#endif

namespace tsl {

// Since April 2023, tensorflow::Status is an alias to absl::Status. The first
// TF release including this change will be TF 2.14 (the latest release in
// April 2023 is 2.13).
// At the same time `tsl::errors::Code` aliases `absl::StatusCode`.
//
// Here is a set of correspondences:
// - Use `absl::OkStatus()` instead of `tsl::OkStatus()`.
typedef absl::Status Status ABSL_DEPRECATE_AND_INLINE();

namespace errors {
typedef absl::StatusCode Code ABSL_DEPRECATE_AND_INLINE();
}  // namespace errors
namespace error {
typedef ::tensorflow::error::Code Code;
}  // namespace error
}  // namespace tsl

// Transparent comparison between tensorflow::error::Code protobuf enum and
// absl::Status.
//
// The longer term objective is to delete these when we have done the transition
// to absl::Status.
namespace tensorflow::error {
inline bool operator==(const ::tensorflow::error::Code& c1,
                       const absl::StatusCode& c2) {
  return static_cast<int>(c1) == static_cast<int>(c2);
}

inline bool operator!=(const ::tensorflow::error::Code& c1,
                       const absl::StatusCode& c2) {
  return static_cast<int>(c1) != static_cast<int>(c2);
}
}  // namespace tensorflow::error

namespace absl {
inline bool operator==(const ::absl::StatusCode& c1,
                       const ::tensorflow::error::Code& c2) {
  return static_cast<int>(c1) == static_cast<int>(c2);
}

inline bool operator!=(const ::absl::StatusCode& c1,
                       const ::tensorflow::error::Code& c2) {
  return static_cast<int>(c1) != static_cast<int>(c2);
}
}  // namespace absl

namespace tsl {

// OkStatus()
//
// Returns an OK status, equivalent to a default constructed instance. Prefer
// usage of `OkStatus()` when constructing such an OK status.
ABSL_DEPRECATE_AND_INLINE() inline absl::Status OkStatus() {
  return absl::OkStatus();
}

// TODO(b/197552541) Move this namespace to errors.h.
namespace errors {

void SetStackTrace(absl::Status& status, std::vector<StackFrame> stack_trace);

std::vector<StackFrame> GetStackTrace(const absl::Status& status);
}  // namespace errors

// Helper class to manage multiple child status values.
class StatusGroup {
 public:
  StatusGroup();
  // Constructor to form a StatusGroup from any N set of Status arguments.
  // Usage: StatusGroup({status_a, status_b, status_c});
  StatusGroup(std::initializer_list<absl::Status> statuses);

  // Utility function to mark a Status as derived. By marking derived status,
  // Derived status messages are ignored when reporting errors to end users.
  static absl::Status MakeDerived(const absl::Status& s);
  static bool IsDerived(const absl::Status& s);

  // Enable warning and error log collection for appending to the aggregated
  // status. This function may be called more than once.
  static void ConfigureLogHistory();

  // Returns merged payloads of all statuses. In case multiple statuses have the
  // same payload key, non-derived statuses have priority over derived ones,
  // otherwise one payload value will be chosen in an unspecified but
  // deterministic order.
  // NOTE: The payload marking derived statuses as derived will not be returned.
  std::unordered_map<std::string, absl::Cord> GetPayloads() const;

  // Return a merged status with combined child status messages with a summary.
  absl::Status as_summary_status() const;
  // Return a merged status with combined child status messages with
  // concatenation.
  absl::Status as_concatenated_status() const;

  bool ok() const { return ok_; }

  // Augment this group with the child status `status`.
  void Update(const absl::Status& status);

  // Attach recent warning and error log messages
  void AttachLogMessages();
  bool HasLogMessages() const { return !recent_logs_.empty(); }

 private:
  bool ok_ = true;
  size_t num_ok_ = 0;

  // Maintain a sorted collection of statuses.
  struct CompareStatus {
    bool operator()(const absl::Status& a, const absl::Status& b) const {
      return a.ToString() > b.ToString();
    }
  };
  // Using std::set instead of absl::btree_set to keep size for certain
  // dependent libraries under the limit.
  std::set<absl::Status, CompareStatus> derived_;
  std::set<absl::Status, CompareStatus> non_derived_;

  std::vector<std::string> recent_logs_;  // recent warning and error logs
};

typedef std::function<void(const absl::Status&)> StatusCallback;

#ifdef SWIG
#define TF_CHECK_OK(val) CHECK_OK(val)
#define TF_QCHECK_OK(val) QCHECK_OK(val)
#define TF_DCHECK_OK(val) DCHECK_OK(val)
#else

ABSL_DEPRECATED("TF_CHECK_OK macro is deprecated. call CHECK_OK instead")
inline void TfCheckOkDeprecationMarker() {}

#define TF_CHECK_OK(val) CHECK_OK((::tsl::TfCheckOkDeprecationMarker(), val))
#define TF_QCHECK_OK(val) QCHECK_OK((::tsl::TfCheckOkDeprecationMarker(), val))
#define TF_DCHECK_OK(val) DCHECK_OK((::tsl::TfCheckOkDeprecationMarker(), val))

#endif

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_STATUS_H_
