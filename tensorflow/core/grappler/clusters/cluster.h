/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_CLUSTERS_CLUSTER_H_
#define TENSORFLOW_CORE_GRAPPLER_CLUSTERS_CLUSTER_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace grappler {

// A cluster represents of collection of hardware resources available to run
// the TensorFlow model.
// A process can only create a single cluster at a time.
class Cluster {
 public:
  explicit Cluster(int timeout_s);
  virtual ~Cluster();

  // Returns a string that represent the type of cluster that was instantiated.
  virtual string type() const = 0;

  // Provision the hardware resources needed to run TensorFlow and start a
  // TensorFlow session that can take advantage of these resources.
  // The actual resources that are leveraged depend on the type of cluster
  // instantiated.
  // Returns OK iff all the requested resources could be reserved and a
  // TensorFlow session successfully created. Returns an error otherwise.
  // There is no graceful degradation to handle the case where only a subset
  // of the requested resources are available.
  virtual Status Provision() = 0;

  // Attempts to shutdown the cluster.
  // Returns OK iff there are no pending calls to the Run() method and all the
  // resources used by the cluster could be released. Returns an error
  // otherwise.
  virtual Status Shutdown() { return Status::OK(); }

  // Whether soft placement is allowed. If allow_soft_placement is true,
  // an op will be placed on CPU if there's no GPU implementation for the OP
  // or if no GPU devices are known or registered or if we need to co-locate
  // with reftype input(s) which are from CPU.
  void AllowSoftPlacement(bool soft_placement_state);

  // Update the number of inter-op threads for each per-session threadpool
  void SetNumInterOpThreads(int num_threads);

  // Set the number of steps required to warmup TensorFlow. Must be called
  // before Provision().
  void SetNumWarmupSteps(int num_steps);

  // Set executor type to instantiate
  void SetExecutorType(const string* executor_type);

  // Returns the number of warmup steps.
  int NumWarmupSteps() const;

  // Disable the collection of detailed statistics. Must be called
  // before Provision().
  void DisableDetailedStats(bool disable);

  // Returns true iff the collection of detailed statistics is enabled.
  bool DetailedStatsEnabled() const;

  // Disable the TensorFlow optimizer. This ensures that the graph that TF
  // executes is similar to the input graph. Must be called before Provision().
  void DisableOptimizer(bool disable);

  // Return the list of TensorFlow devices that are available to execute a
  // graph. This is empty until provision() is called.
  const std::unordered_map<string, DeviceProperties>& GetDevices() const {
    return devices_;
  }

  // Convenience method that returns the set of device names. These names are
  // sorted alphabetically.
  const std::vector<string> GetDeviceNames() const;

  // The DeviceSet is not always available, but when it is it contains a
  // superset of the devices listed in GetDevices/GetDeviceNames().
  virtual const DeviceSet* GetDeviceSet() const { return nullptr; }

  // Enables collecting the allocator stats. If called, must be called before
  // Provision().
  virtual Status EnablePeakMemoryStats() {
    return errors::Unimplemented(strings ::StrCat(
        "Peak Memory Stats are not supported on ", type(), " clusters"));
  }

  // Returns peak memory of all devices during the session creation and session
  // runs.
  virtual Status GetPeakMemoryUsage(
      std::unordered_map<string, uint64>* device_peak_memory) const {
    return errors::Unimplemented(
        "GetPeakMemoryUsage is not implemented for this type of cluster.");
  }

  // Prepare the session to run the specified grappler item. This include
  // initializing all the model variables.
  virtual Status Initialize(const GrapplerItem& item) = 0;

  // Run the specified graph_def and return the corresponding metadata.
  virtual Status Run(const GraphDef& graph_def,
                     const std::vector<std::pair<string, Tensor>>& feed,
                     const std::vector<string>& fetch,
                     RunMetadata* metadata) = 0;

  // Run the specified GrapplerItem and return the corresponding metadata.
  virtual Status Run(const GrapplerItem& item, RunMetadata* metadata) {
    return Run(item.graph, item.feed, item.fetch, metadata);
  }

 protected:
  std::unordered_map<string, DeviceProperties> devices_;
  const int timeout_s_;
  SessionOptions options_;
  RunOptions run_options_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_CLUSTERS_CLUSTER_H_
