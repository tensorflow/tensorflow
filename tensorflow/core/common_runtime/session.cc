/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <string>

#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

namespace {
Status GetFactory(const SessionOptions& options, SessionFactory** ret) {
  string runtime_type = "DIRECT_SESSION";
  if (!options.target.empty()) {
    // Use the service based session.
    runtime_type = "REMOTE_SESSION";
  }
  *ret = SessionFactory::GetFactory(runtime_type);
  if (!*ret) {
    return errors::NotFound("Could not find session factory for ",
                            runtime_type);
  }
  return Status::OK();
}
}  // end namespace

Session* NewSession(const SessionOptions& options) {
  SessionFactory* factory;
  Status s = GetFactory(options, &factory);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return nullptr;
  }
  return factory->NewSession(options);
}

Status NewSession(const SessionOptions& options, Session** out_session) {
  SessionFactory* factory;
  Status s = GetFactory(options, &factory);
  if (!s.ok()) {
    *out_session = nullptr;
    LOG(ERROR) << s;
    return s;
  }
  *out_session = factory->NewSession(options);
  if (!*out_session) {
    return errors::Internal("Failed to create session.");
  }
  return Status::OK();
}

}  // namespace tensorflow
