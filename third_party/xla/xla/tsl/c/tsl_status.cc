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

#include "xla/tsl/c/tsl_status.h"

#include <string>

#include "xla/tsl/c/tsl_status_internal.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

using ::tsl::Status;
using ::tsl::error::Code;
using ::tsl::errors::IOError;

TSL_Status* TSL_NewStatus() { return new TSL_Status; }

void TSL_DeleteStatus(TSL_Status* s) { delete s; }

void TSL_SetStatus(TSL_Status* s, TSL_Code code, const char* msg) {
  if (code == TSL_OK) {
    s->status = absl::OkStatus();
    return;
  }
  s->status =
      Status(static_cast<absl::StatusCode>(code), tsl::StringPiece(msg));
}

void TSL_SetPayload(TSL_Status* s, const char* key, const char* value) {
  s->status.SetPayload(key, absl::Cord(absl::string_view(value)));
}

void TSL_ForEachPayload(const TSL_Status* s, TSL_PayloadVisitor visitor,
                        void* capture) {
  s->status.ForEachPayload([visitor, capture](absl::string_view type_url,
                                              const absl::Cord& payload) {
    std::string type_url_str(type_url);
    std::string payload_str(payload);
    visitor(type_url_str.c_str(), payload_str.c_str(), capture);
  });
}

void TSL_SetStatusFromIOError(TSL_Status* s, int error_code,
                              const char* context) {
  // TODO(b/139060984): Handle windows when changing its filesystem
  s->status = IOError(context, error_code);
}

TSL_Code TSL_GetCode(const TSL_Status* s) {
  return static_cast<TSL_Code>(s->status.code());
}

const char* TSL_Message(const TSL_Status* s) {
  return absl::StatusMessageAsCStr(s->status);
}
