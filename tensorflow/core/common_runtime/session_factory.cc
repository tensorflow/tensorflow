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

#include "tensorflow/core/common_runtime/session_factory.h"

#include <unordered_map>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

static mutex* get_session_factory_lock() {
  static mutex session_factory_lock(LINKER_INITIALIZED);
  return &session_factory_lock;
}

typedef std::unordered_map<string, SessionFactory*> SessionFactories;
SessionFactories* session_factories() {
  static SessionFactories* factories = new SessionFactories;
  return factories;
}

}  // namespace

void SessionFactory::Register(const string& runtime_type,
                              SessionFactory* factory) {
  mutex_lock l(*get_session_factory_lock());
  if (!session_factories()->insert({runtime_type, factory}).second) {
    LOG(ERROR) << "Two session factories are being registered "
               << "under " << runtime_type;
  }
}

namespace {
const string RegisteredFactoriesErrorMessageLocked() {
  std::vector<string> factory_types;
  for (const auto& session_factory : *session_factories()) {
    factory_types.push_back(session_factory.first);
  }
  return strings::StrCat("Registered factories are {",
                         absl::StrJoin(factory_types, ", "), "}.");
}
string SessionOptionsToString(const SessionOptions& options) {
  return strings::StrCat("target: \"", options.target,
                         "\" config: ", options.config.ShortDebugString());
}
}  // namespace

absl::Status SessionFactory::GetFactory(const SessionOptions& options,
                                        SessionFactory** out_factory) {
  mutex_lock l(*get_session_factory_lock());  // could use reader lock

  std::vector<std::pair<string, SessionFactory*>> candidate_factories;
  for (const auto& session_factory : *session_factories()) {
    if (session_factory.second->AcceptsOptions(options)) {
      VLOG(2) << "SessionFactory type " << session_factory.first
              << " accepts target: " << options.target;
      candidate_factories.push_back(session_factory);
    } else {
      VLOG(2) << "SessionFactory type " << session_factory.first
              << " does not accept target: " << options.target;
    }
  }

  if (candidate_factories.size() == 1) {
    *out_factory = candidate_factories[0].second;
    return absl::OkStatus();
  } else if (candidate_factories.size() > 1) {
    // NOTE(mrry): This implementation assumes that the domains (in
    // terms of acceptable SessionOptions) of the registered
    // SessionFactory implementations do not overlap. This is fine for
    // now, but we may need an additional way of distinguishing
    // different runtimes (such as an additional session option) if
    // the number of sessions grows.
    // TODO(mrry): Consider providing a system-default fallback option
    // in this case.
    std::vector<string> factory_types;
    factory_types.reserve(candidate_factories.size());
    for (const auto& candidate_factory : candidate_factories) {
      factory_types.push_back(candidate_factory.first);
    }
    return errors::Internal(
        "Multiple session factories registered for the given session "
        "options: {",
        SessionOptionsToString(options), "} Candidate factories are {",
        absl::StrJoin(factory_types, ", "), "}. ",
        RegisteredFactoriesErrorMessageLocked());
  } else {
    return errors::NotFound(
        "No session factory registered for the given session options: {",
        SessionOptionsToString(options), "} ",
        RegisteredFactoriesErrorMessageLocked());
  }
}

}  // namespace tensorflow
