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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_HIERARCHICAL_TREE_BROADCASTER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_HIERARCHICAL_TREE_BROADCASTER_H_

#include <vector>

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/framework/collective.h"

namespace tensorflow {

// Hierarchical tree-algorithm implementation of collective broadcast.
class HierarchicalTreeBroadcaster : public CollectiveImplementationInterface {
 public:
  HierarchicalTreeBroadcaster();
  ~HierarchicalTreeBroadcaster() override = default;

  // Establishes the subdiv permutations needed for a hierarchical broadcast.
  // If all devices are local, establishes a single subdiv comprising all
  // devices.  If any devices are on a different task, establishes n+1 subdivs
  // for n tasks.
  // The first subdiv comprises one device per task which gets the tensor on
  // each task.  Subdiv i+1 corresponds to a task-local tree-broadcast for task
  // i.
  Status InitializeCollectiveParams(CollectiveParams* col_params) override;

  // Initializes members of CollectiveContext not yet initialized, i.e. device
  // and device_locality.  Also saves the CollectiveContext in this object.
  Status InitializeCollectiveContext(CollectiveContext* col_ctx) override;

  // No-op for hierarchical tree broadcaster.
  Status InitializeInstanceBeforeGroupDiscovery(CollectiveParams*) override {
    return Status::OK();
  }

  // Begins async execution of the hierarchical tree broadcast.
  // Must be called in a blockable thread.
  // TODO(b/80529858): remove the previous warning when we have a dedicated
  // collective threadpool.
  void Run(StatusCallback done) override;

  // Returns the rank of the device from which this device should receive
  // its value, -1 if no value should be received.
  static int TreeRecvFrom(const CollectiveParams& cp, int subdiv);

  // Populates targets with the ranks of the devices to which this device
  // should forward the value.
  static void TreeSendTo(const CollectiveParams& cp, int subdiv,
                         std::vector<int>* targets);

 private:
  // Get the task to which the device at `device_rank` belongs.
  int GetDeviceTask(int device_rank, const std::vector<int>& dev_per_task);

  // Sends `src_tensor` asynchronously from this device to device at `dst_rank`
  // in `subdiv`.  Calls `done` upon completion.
  void DispatchSend(int subdiv, int dst_rank, int src_rank,
                    const Tensor* src_tensor, const StatusCallback& done);

  // Receives a tensor into the memory buffer owned by `dst_tensor` at this
  // device from device at `src_rank` in `subdiv`.  Calls `done` upon
  // completion.
  void DispatchRecv(int subdiv, int src_rank, int dst_rank, Tensor* dst_tensor,
                    const StatusCallback& done);

  // Executes the hierarchical broadcast defined by this op.
  void RunTree();

  CollectiveContext* col_ctx_;          // Not owned
  const CollectiveParams* col_params_;  // Not owned
  StatusCallback done_;
  Status status_;
  bool is_source_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_HIERARCHICAL_TREE_BROADCASTER_H_
