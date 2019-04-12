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

#include "tensorflow/core/lib/core/status.h"
#include <stdio.h>
#include <map>
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {

Status::Status(tensorflow::error::Code code, StringPiece msg) {
  assert(code != tensorflow::error::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = string(msg);
}

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const string& Status::empty_string() {
  static string* empty = new string;
  return *empty;
}

string error_name(error::Code code) {
  switch (code) {
    case tensorflow::error::OK:
      return "OK";
      break;
    case tensorflow::error::CANCELLED:
      return "Cancelled";
      break;
    case tensorflow::error::UNKNOWN:
      return "Unknown";
      break;
    case tensorflow::error::INVALID_ARGUMENT:
      return "Invalid argument";
      break;
    case tensorflow::error::DEADLINE_EXCEEDED:
      return "Deadline exceeded";
      break;
    case tensorflow::error::NOT_FOUND:
      return "Not found";
      break;
    case tensorflow::error::ALREADY_EXISTS:
      return "Already exists";
      break;
    case tensorflow::error::PERMISSION_DENIED:
      return "Permission denied";
      break;
    case tensorflow::error::UNAUTHENTICATED:
      return "Unauthenticated";
      break;
    case tensorflow::error::RESOURCE_EXHAUSTED:
      return "Resource exhausted";
      break;
    case tensorflow::error::FAILED_PRECONDITION:
      return "Failed precondition";
      break;
    case tensorflow::error::ABORTED:
      return "Aborted";
      break;
    case tensorflow::error::OUT_OF_RANGE:
      return "Out of range";
      break;
    case tensorflow::error::UNIMPLEMENTED:
      return "Unimplemented";
      break;
    case tensorflow::error::INTERNAL:
      return "Internal";
      break;
    case tensorflow::error::UNAVAILABLE:
      return "Unavailable";
      break;
    case tensorflow::error::DATA_LOSS:
      return "Data loss";
      break;
    default:
      char tmp[30];
      snprintf(tmp, sizeof(tmp), "Unknown code(%d)", static_cast<int>(code));
      return tmp;
      break;
  }
}

string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    string result(error_name(code()));
    result += ": ";
    result += state_->msg;
    return result;
  }
}

void Status::IgnoreError() const {
  // no-op
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

string* TfCheckOpHelperOutOfLine(const ::tensorflow::Status& v,
                                 const char* msg) {
  string r("Non-OK-status: ");
  r += msg;
  r += " status: ";
  r += v.ToString();
  // Leaks string but this is only to be used in a fatal error message
  return new string(r);
}

// kDerivedMarker is appended to the Status message string to indicate whether a
// Status object is the root cause of an error or if it has been triggered by
// cancelling/aborting a step.
static const char* kDerivedMarker = "[_Derived_]";

Status StatusGroup::MakeDerived(const Status& s) {
  return Status(s.code(), strings::StrCat(kDerivedMarker, s.error_message()));
}

bool StatusGroup::IsDerived(const Status& s) {
  return s.error_message().find(kDerivedMarker) != std::string::npos;
}

void StatusGroup::Update(const Status& s) {
  if (s.ok()) {
    ++num_ok_;
  } else {
    ok_ = false;
    children_.push_back(s);
  }
}

static std::vector<Status> GetNonDerivedStatuses(
    const std::vector<Status>& status) {
  std::vector<Status> nonderived_statuses;
  for (auto& s : status) {
    if (!StatusGroup::IsDerived(s)) {
      nonderived_statuses.push_back(s);
    }
  }
  return nonderived_statuses;
}

static constexpr int kMaxAggregatedStatusMessageSize = 8 * 1024;

// Summarize all the status objects in the StatusGroup. This is used when
// individual Status objects in the StatusGroup are not already summarized.
Status StatusGroup::as_summary_status() const {
  if (ok_) {
    return Status::OK();
  }

  std::vector<Status> nonderived_statuses = GetNonDerivedStatuses(children_);

  // If only one root status is found, return it directly.
  if (nonderived_statuses.size() == 1) {
    return nonderived_statuses[0];
  }

  if (!nonderived_statuses.empty()) {
    std::vector<string> fmt;

    fmt.push_back(strings::Printf("%zu root error(s) found.",
                                  nonderived_statuses.size()));

    int index = 0;
    for (auto& s : nonderived_statuses) {
      fmt.emplace_back(strings::StrCat("  (", index, ") ", s.ToString()));
      ++index;
    }

    fmt.push_back(strings::Printf("%zu successful operations.", num_ok_));
    fmt.push_back(
        strings::Printf("%zu derived errors ignored.",
                        children_.size() - nonderived_statuses.size()));
    return Status(
        nonderived_statuses[0].code(),
        absl::StrJoin(fmt, "\n").substr(0, kMaxAggregatedStatusMessageSize));
  } else {
    // All status are derived. Return the first status error, which will be
    // ignored when aggregated at the master node.
    return children_[0];
  }
}

// Concatenate all the status objects in the StatusGroup. This is used when
// individual Status objects in the StatusGroup are already summarized Status.
Status StatusGroup::as_concatenated_status() const {
  if (ok_) {
    return Status::OK();
  }

  std::vector<Status> nonderived_statuses = GetNonDerivedStatuses(children_);

  // If only one root status is found, return it directly.
  if (nonderived_statuses.size() == 1) {
    return nonderived_statuses[0];
  }

  if (!nonderived_statuses.empty()) {
    std::vector<string> fmt;
    fmt.emplace_back("\n=====================");
    for (auto& s : nonderived_statuses) {
      fmt.emplace_back(s.ToString());
    }
    fmt.emplace_back("=====================\n");
    return Status(
        nonderived_statuses[0].code(),
        absl::StrJoin(fmt, "\n").substr(0, kMaxAggregatedStatusMessageSize));
  } else {
    // All statuses are derived. Pick the first available status to return.
    // This should not happen in normal execution.
    return children_[0];
  }
}

}  // namespace tensorflow
