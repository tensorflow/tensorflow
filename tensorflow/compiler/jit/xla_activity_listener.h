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
#ifndef TENSORFLOW_COMPILER_JIT_XLA_ACTIVITY_LISTENER_H_
#define TENSORFLOW_COMPILER_JIT_XLA_ACTIVITY_LISTENER_H_

#include <memory>

#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
// Broadcast `auto_clustering_activity` to all the registered listeners.
Status BroadcastXlaActivity(XlaAutoClusteringActivity auto_clustering_activity);

// Broadcast `jit_compilation_activity` to all the registered listeners.
Status BroadcastXlaActivity(XlaJitCompilationActivity jit_compilation_activity);

// Various components of the system can subclass XlaActivityListener to
// notifications on auto-clustering and JIT compilation events.
//
// Subclasses of XlaActivityListener must be thread safe.
class XlaActivityListener {
 public:
  // Called after TensorFlow auto-clusters a graph.
  virtual Status Listen(
      const XlaAutoClusteringActivity& auto_clustering_activity) = 0;

  // Called after TensorFlow JIT compiles an XLA cluster.
  virtual Status Listen(
      const XlaJitCompilationActivity& jit_compilation_activity) = 0;

  virtual ~XlaActivityListener();
};

// Registers an `XlaActivityListener`, which will be invoked on all subsequent
// `BroadcastXlaActivity` calls.
void RegisterXlaActivityListener(std::unique_ptr<XlaActivityListener> listener);

using GlobalProcessIdMaker = std::function<std::string()>;

// Installs `global_process_id_maker` as a "global process id" maker.
//
// The value returned by the global process ID maker, if one is installed, is
// stored in the global_process_id field of the Xla*Activity messages before
// they're fed to the registered activity listeners.  If no ID maker is
// installed then global_process_id is set to "unknown".
//
// `global_process_id_maker` must be thread safe.
//
// The global process id maker is used to tag *Activity messages to so that the
// broadcasting process can be uniquely identified.  Therefore the global
// process id maker
//
//  - Must always return the same value within the same process.
//  - Cannot be set or changed after we have broadcasted any XLA activity.
void SetGlobalProcessIdMaker(GlobalProcessIdMaker global_process_id_maker);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_ACTIVITY_LISTENER_H_
