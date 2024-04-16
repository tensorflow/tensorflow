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

#ifndef TENSORFLOW_TSL_PLATFORM_STATUS_H_
#define TENSORFLOW_TSL_PLATFORM_STATUS_H_

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
#include "absl/types/optional.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/stack_frame.h"
#include "tsl/platform/types.h"
#include "tsl/protobuf/error_codes.pb.h"

// Include appropriate platform-dependent parts of status.
#if defined(PLATFORM_GOOGLE)
#include "tsl/platform/google/status.h"  // IWYU pragma: export
#else
#include "tsl/platform/default/status.h"  // IWYU pragma: export
#endif

// TODO: b/323943471 - This macro should eventually be provided by Abseil.
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
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
};

ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FromAbslStatus(const absl::Status& s) { return s; }
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ToAbslStatus(const ::absl::Status& s) { return s; }

// Given `Status.message()` does not guarantee to be always backed by a
// null-terminated string, we have this utility function when it's needed for
// the Tensorflow C-API.
// A more robust API would be to get both a `char*` of the beginning of the
// string, plus the size (see e.g. `XlaCustomCallStatusSetFailure`).
const char* NullTerminatedMessage(const Status& status);

// TODO(b/197552541) Move this namespace to errors.h.
namespace errors {

void SetStackTrace(::tsl::Status& status, std::vector<StackFrame> stack_trace);

std::vector<StackFrame> GetStackTrace(const ::tsl::Status& status);
}  // namespace errors

// Helper class to manage multiple child status values.
class StatusGroup {
 public:
  StatusGroup();
  // Constructor to form a StatusGroup from any N set of Status arguments.
  // Usage: StatusGroup({status_a, status_b, status_c});
  StatusGroup(std::initializer_list<Status> statuses);

  // Utility function to mark a Status as derived. By marking derived status,
  // Derived status messages are ignored when reporting errors to end users.
  static Status MakeDerived(const Status& s);
  static bool IsDerived(const Status& s);

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
  Status as_summary_status() const;
  // Return a merged status with combined child status messages with
  // concatenation.
  Status as_concatenated_status() const;

  bool ok() const { return ok_; }

  // Augment this group with the child status `status`.
  void Update(const Status& status);

  // Attach recent warning and error log messages
  void AttachLogMessages();
  bool HasLogMessages() const { return !recent_logs_.empty(); }

 private:
  bool ok_ = true;
  size_t num_ok_ = 0;

  // Maintain a sorted collection of statuses.
  struct CompareStatus {
    bool operator()(const Status& a, const Status& b) const {
      return a.ToString() > b.ToString();
    }
  };
  // Using std::set instead of absl::btree_set to keep size for certain
  // dependent libraries under the limit.
  std::set<Status, CompareStatus> derived_;
  std::set<Status, CompareStatus> non_derived_;

  std::vector<std::string> recent_logs_;  // recent warning and error logs
};


typedef std::function<void(const Status&)> StatusCallback;

extern ::tsl::string* TfCheckOpHelperOutOfLine(const ::tsl::Status& v,
                                               const char* msg);

inline ::tsl::string* TfCheckOpHelper(::tsl::Status v, const char* msg) {
  if (v.ok()) return nullptr;
  return TfCheckOpHelperOutOfLine(v, msg);
}

#define TF_DO_CHECK_OK(val, level)                          \
  while (auto* _result = ::tsl::TfCheckOpHelper(val, #val)) \
  LOG(level) << *(_result)

#define TF_CHECK_OK(val) TF_DO_CHECK_OK(val, FATAL)
#define TF_QCHECK_OK(val) TF_DO_CHECK_OK(val, QFATAL)

// DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
  while (false && (::tsl::OkStatus() == (val))) LOG(FATAL)
#endif

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_STATUS_H_
