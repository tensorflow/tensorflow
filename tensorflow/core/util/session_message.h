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

#ifndef TENSORFLOW_CORE_UTIL_SESSION_MESSAGE_H_
#define TENSORFLOW_CORE_UTIL_SESSION_MESSAGE_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

class ResourceMgr;
class SessionStatus;

class SessionLogger : public ResourceBase {
 public:
  SessionLogger();
  ~SessionLogger();

  void Log(StringPiece message);
  string DebugString() override;

  const SessionStatus& status() { return *status_; }

 private:
  std::unique_ptr<SessionStatus> status_;
  mutex mu_;
};

// Return a SessionLogger instance for the current session.  If the logger
// will be used across multiple computations, you must explicitly acquire
// and release references using Ref()/Unref().
//
// Returns nullptr if a logger cannot be created.
SessionLogger* GetSessionLogger(ResourceMgr* rm);

// Attach `message` to the logger for the current session.
void LogSessionMessage(ResourceMgr* rm, StringPiece message);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_SESSION_MESSAGE_H
