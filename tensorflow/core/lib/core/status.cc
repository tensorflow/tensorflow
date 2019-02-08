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

void StatusGroup::Update(const Status& s) {
  if (s.ok()) {
    ++num_ok_;
  } else {
    ok_ = false;
    children_.push_back(s);
  }
}

const int kMaxChildMessageSize = 2048;

Status StatusGroup::as_status() const {
  if (ok_) {
    return Status::OK();
  }

  // Reduce verbosity when handling duplicate messages. If there is only a
  // single message, or all messages have similar content, then return the
  // longest status message.
  std::vector<Status> sorted_children(children_);
  std::sort(sorted_children.begin(), sorted_children.end(),
            [](const Status& a, const Status& b) {
              return a.error_message().length() > b.error_message().length();
            });
  bool single_status = true;
  for (const auto& s : sorted_children) {
    if (s.code() != sorted_children[0].code() ||
        sorted_children[0].error_message().find(s.error_message()) ==
            string::npos) {
      single_status = false;
      break;
    }
  }

  if (single_status) {
    return sorted_children[0];
  }

  std::vector<string> fmt;

  // Compute a final output string with status codes sorted by frequency in
  // increasing order.  This prefers more "interesting" messages over child
  // messages that may come from cancellation.
  std::map<error::Code, std::vector<Status>> code_to_status;
  for (const Status& s : children_) {
    code_to_status[s.code()].push_back(s);
  }

  std::vector<std::pair<error::Code, int>> count_vec;
  count_vec.reserve(code_to_status.size());
  for (auto& p : code_to_status) {
    count_vec.push_back(std::make_pair(p.first, p.second.size()));
  }

  std::sort(
      count_vec.begin(), count_vec.end(),
      [](const std::pair<error::Code, int>& a,
         const std::pair<error::Code, int>& b) { return a.second < b.second; });

  fmt.push_back(
      strings::Printf("Combined status information from %zu operations:\n",
                      num_ok_ + children_.size()));

  for (const auto& p : count_vec) {
    // Deduplicate error messages
    std::map<string, int> child_errors;
    for (const Status& s : code_to_status[p.first]) {
      ++child_errors[s.error_message()];
    }

    string child_fmt;
    for (auto& m : child_errors) {
      child_fmt.append(strings::Printf(
          "  %s [%dx]",
          str_util::StringReplace(m.first, "\n", "\n  ", true).c_str(),
          m.second));
      child_fmt.append("\n");
    }
    // Strip last newline.
    child_fmt = child_fmt.substr(0, child_fmt.size() - 1);

    if (child_fmt.size() > kMaxChildMessageSize) {
      child_fmt =
          strings::StrCat(child_fmt.substr(0, kMaxChildMessageSize), "...");
    }
    fmt.push_back(strings::Printf("Status code: %s [%dx]\n%s",
                                  error_name(p.first).c_str(), p.second,
                                  child_fmt.c_str()));
  }

  fmt.push_back(strings::Printf("(%zd successful operations.)", num_ok_));

  // TODO(power): use the least-frequently occurring status for the return code
  return Status(children_[0].code(), str_util::Join(fmt, "\n"));
}

}  // namespace tensorflow
