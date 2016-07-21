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

namespace tensorflow {

Status::Status(tensorflow::error::Code code, StringPiece msg) {
  assert(code != tensorflow::error::OK);
  state_ = new State;
  state_->code = code;
  state_->msg = msg.ToString();
}

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* src) {
  delete state_;
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = new State(*src);
  }
}

const string& Status::empty_string() {
  static string* empty = new string;
  return *empty;
}

string Status::ToString() const {
  if (state_ == NULL) {
    return "OK";
  } else {
    char tmp[30];
    const char* type;
    switch (code()) {
      case tensorflow::error::CANCELLED:
        type = "Cancelled";
        break;
      case tensorflow::error::UNKNOWN:
        type = "Unknown";
        break;
      case tensorflow::error::INVALID_ARGUMENT:
        type = "Invalid argument";
        break;
      case tensorflow::error::DEADLINE_EXCEEDED:
        type = "Deadline exceeded";
        break;
      case tensorflow::error::NOT_FOUND:
        type = "Not found";
        break;
      case tensorflow::error::ALREADY_EXISTS:
        type = "Already exists";
        break;
      case tensorflow::error::PERMISSION_DENIED:
        type = "Permission denied";
        break;
      case tensorflow::error::UNAUTHENTICATED:
        type = "Unauthenticated";
        break;
      case tensorflow::error::RESOURCE_EXHAUSTED:
        type = "Resource exhausted";
        break;
      case tensorflow::error::FAILED_PRECONDITION:
        type = "Failed precondition";
        break;
      case tensorflow::error::ABORTED:
        type = "Aborted";
        break;
      case tensorflow::error::OUT_OF_RANGE:
        type = "Out of range";
        break;
      case tensorflow::error::UNIMPLEMENTED:
        type = "Unimplemented";
        break;
      case tensorflow::error::INTERNAL:
        type = "Internal";
        break;
      case tensorflow::error::UNAVAILABLE:
        type = "Unavailable";
        break;
      case tensorflow::error::DATA_LOSS:
        type = "Data loss";
        break;
      default:
        snprintf(tmp, sizeof(tmp), "Unknown code(%d)",
                 static_cast<int>(code()));
        type = tmp;
        break;
    }
    string result(type);
    result += ": ";
    result += state_->msg;
    return result;
  }
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

}  // namespace tensorflow
