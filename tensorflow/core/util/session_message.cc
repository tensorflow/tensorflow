/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/session_message.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/event.pb.h"

static const int kMaxLogEvents = 1000;

namespace tensorflow {

SessionLogger::SessionLogger() : status_(new SessionStatus) {}

SessionLogger::~SessionLogger() {}

string SessionLogger::DebugString() { return "SessionLogger"; }

void SessionLogger::Log(StringPiece message) {
  mutex_lock lock(mu_);

  Event* event = status_->add_event();
  event->set_wall_time(Env::Default()->NowMicros());
  event->set_step(0);
  LogMessage* log = event->mutable_log_message();
  log->set_message(message.ToString());
  log->set_level(LogMessage::INFO);

  // Clip log events by 10% if we overflow
  if (status_->event_size() > kMaxLogEvents) {
    auto events = status_->mutable_event();
    events->DeleteSubrange(0, kMaxLogEvents / 10);
  }
}

SessionLogger* GetSessionLogger(ResourceMgr* rm) {
  SessionLogger* logger;

  std::function<Status(SessionLogger**)> status_creator =
      [](SessionLogger** result) {
        *result = new SessionLogger();
        return Status::OK();
      };

  if (!rm->LookupOrCreate<SessionLogger>("session", "status", &logger,
                                         status_creator)
           .ok()) {
    return nullptr;
  }

  return logger;
}

void LogSessionMessage(ResourceMgr* rm, StringPiece message) {
  return GetSessionLogger(rm)->Log(message);
}

}  // namespace tensorflow
