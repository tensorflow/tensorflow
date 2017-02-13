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

#ifndef TENSORFLOW_CORE_LIB_CORE_STATUS_H_
#define TENSORFLOW_CORE_LIB_CORE_STATUS_H_

#include <functional>
#include <iosfwd>
#include <string>
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

/// @ingroup core
/// Denotes success or failure of a call in Tensorflow.
class Status {
 public:
  /// Create a success status.
  Status() : state_(NULL) {}
  ~Status() { delete state_; }

  /// \brief Create a status with the specified error code and msg as a
  /// human-readable string containing more detailed information.
  Status(tensorflow::error::Code code, tensorflow::StringPiece msg);

  /// Copy the specified status.
  Status(const Status& s);
  void operator=(const Status& s);

  static Status OK() { return Status(); }

  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == NULL); }

  tensorflow::error::Code code() const {
    return ok() ? tensorflow::error::OK : state_->code;
  }

  const string& error_message() const {
    return ok() ? empty_string() : state_->msg;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status& new_status);

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

 private:
  static const string& empty_string();
  struct State {
    tensorflow::error::Code code;
    string msg;
  };
  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  State* state_;

  void SlowCopyFrom(const State* src);
};

inline Status::Status(const Status& s)
    : state_((s.state_ == NULL) ? NULL : new State(*s.state_)) {}

inline void Status::operator=(const Status& s) {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_);
  }
}

inline bool Status::operator==(const Status& x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

typedef std::function<void(const Status&)> StatusCallback;

#define TF_CHECK_OK(val) CHECK_EQ(::tensorflow::Status::OK(), (val))
#define TF_QCHECK_OK(val) QCHECK_EQ(::tensorflow::Status::OK(), (val))

// DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
  while (false && (::tensorflow::Status::OK() == (val))) LOG(FATAL)
#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_STATUS_H_
