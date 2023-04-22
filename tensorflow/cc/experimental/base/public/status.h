/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_STATUS_H_
#define TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_STATUS_H_

#include <memory>
#include <string>

#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// Status is a wrapper around an error code and an optional error message.
// The set of error codes are defined here:
// https://github.com/tensorflow/tensorflow/blob/08931c1e3e9eb2e26230502d678408e66730826c/tensorflow/c/tf_status.h#L39-L60
// Many Tensorflow APIs return a Status, or take a Status as an out parameter.
// Clients should check for status.ok() after calling these APIs, and either
// handle or propagate the error appropriately.
// TODO(bmzhao): Add a detailed code example before moving out of experimental.
class Status {
 public:
  // Create a success status
  Status() : status_(TF_NewStatus()) {}

  // Return the status code
  TF_Code code() const;

  // Returns the error message in Status.
  std::string message() const;

  // Returns the error message in Status.
  bool ok() const;

  // Record <code, msg> in Status. Any previous information is lost.
  // A common use is to clear a status: SetStatus(TF_OK, "");
  void SetStatus(TF_Code code, const std::string& msg);

  // Status is movable, but not copyable.
  Status(Status&&) = default;
  Status& operator=(Status&&) = default;

 private:
  friend class RuntimeBuilder;
  friend class Runtime;
  friend class SavedModelAPI;
  friend class TensorHandle;

  // Wraps a TF_Status*, and takes ownership of it.
  explicit Status(TF_Status* status) : status_(status) {}

  // Status is not copyable
  Status(const Status&) = delete;
  Status& operator=(const Status&) = delete;

  // Returns the TF_Status that this object wraps. This object
  // retains ownership of the pointer.
  TF_Status* GetTFStatus() const { return status_.get(); }

  struct TFStatusDeleter {
    void operator()(TF_Status* p) const { TF_DeleteStatus(p); }
  };
  std::unique_ptr<TF_Status, TFStatusDeleter> status_;
};

inline TF_Code Status::code() const { return TF_GetCode(status_.get()); }

inline std::string Status::message() const {
  return std::string(TF_Message(status_.get()));
}

inline bool Status::ok() const { return code() == TF_OK; }

inline void Status::SetStatus(TF_Code code, const std::string& msg) {
  TF_SetStatus(status_.get(), code, msg.c_str());
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_STATUS_H_
