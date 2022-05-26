/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/tf_status.h"

#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

using ::tensorflow::Status;
using ::tensorflow::error::Code;
using ::tensorflow::errors::IOError;

TF_Status* TF_NewStatus() { return new TF_Status; }

void TF_DeleteStatus(TF_Status* s) { delete s; }

void TF_SetStatus(TF_Status* s, TF_Code code, const char* msg) {
  if (code == TF_OK) {
    s->status = Status::OK();
    return;
  }
  s->status = Status(static_cast<Code>(code), tensorflow::StringPiece(msg));
}

void TF_SetPayload(TF_Status* s, const char* key, const char* value) {
  s->status.SetPayload(key, absl::Cord(absl::string_view(value)));
}

void TF_SetStatusFromIOError(TF_Status* s, int error_code,
                             const char* context) {
  // TODO(mihaimaruseac): Handle windows when changing its filesystem
  s->status = IOError(context, error_code);
}

TF_Code TF_GetCode(const TF_Status* s) {
  return static_cast<TF_Code>(s->status.code());
}

const char* TF_Message(const TF_Status* s) {
  return s->status.error_message().c_str();
}
