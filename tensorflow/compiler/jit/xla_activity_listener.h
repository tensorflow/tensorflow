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

// Broadcast `jit_compilation_activity` to all the registered listeners.
Status BroadcastOptimizationRemark(XlaOptimizationRemark optimization_remark);

// LINT.IfChange
// Called after TensorFlow realizes possible lost performance. The parameters in
// this should match all of the values in the XlaOptimizationRemark proto.
Status BroadcastOptimizationRemark(
    XlaOptimizationRemark::Warning optimization_warning,
    string debug_information);

// LINT.ThenChange(//tensorflow/compiler/jit/xla_activity.proto)

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

  // Called after TensorFlow realizes possible lost performance.
  virtual Status Listen(const XlaOptimizationRemark& optimization_remark) = 0;

  // Called at program exit in best-effort manner to give listeners a chance to
  // flush their state.
  //
  // Default implementation is a no-op.
  virtual void Flush();

  virtual ~XlaActivityListener();
};

// Registers an `XlaActivityListener`, which will be invoked on all subsequent
// `BroadcastXlaActivity` calls.
void RegisterXlaActivityListener(std::unique_ptr<XlaActivityListener> listener);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_ACTIVITY_LISTENER_H_
