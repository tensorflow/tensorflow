/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_SESSION_INTERFACE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_SESSION_INTERFACE_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

class ThreadPool;

namespace tensorflow {

class CallOptions;
class GraphDef;
class RunStepRequest;
class RunStepResponse;
class ExtendSessionRequest;
class ExtendSessionResponse;

// A "master session" encapsulates a distributed graph computation
// (resource allocation, placement, execution, etc.).
class MasterSessionInterface {
 public:
  // Initializes the Session with "def".  Must be called before Extend(),
  // Run(), or Close().
  //
  // The callee may clear "def".
  virtual Status Create(GraphDef* def) = 0;

  // Returns the session handle.
  virtual const string& handle() const = 0;

  // Returns the last access time (the number of micro-seconds since
  // some fixed point in time) of this session.
  virtual uint64 last_access_time_usec() const = 0;

  // Attempt to extend the graph according to the given "req".
  // (See master.proto for details of valid extensions.)
  //
  // PRECONDITION: The current version of this session's graph
  //   is "req->current_version".
  //
  // POSTCONDITION: The current version of this session's graph
  //   is "req->new_version".
  //
  // Extend() may block the caller thread for a long time.
  virtual Status Extend(const ExtendSessionRequest* req,
                        ExtendSessionResponse* resp) = 0;

  // Run one step.
  virtual Status Run(CallOptions* opts, const RunStepRequest* req,
                     RunStepResponse* resp) = 0;

  // Close this session and delete "*this". Returns OK if all known
  // states are cleanup successfully.
  //
  // Close() may block the caller thread for a long time.
  virtual Status Close() = 0;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_SESSION_INTERFACE_H_
