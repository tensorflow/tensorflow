/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_TEST_CLUSTER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_TEST_CLUSTER_H_

#include "tensorflow/core/data/service/server_lib.h"

namespace tensorflow {
namespace data {

// Helper class for unit testing a tf.data service cluster.
class TestCluster {
 public:
  // Creates a new test cluster with a dispatcher and `num_workers` workers.
  explicit TestCluster(int num_workers);

  // Initializes the test cluster. This must be called before interacting with
  // the cluster. Initialize should be called only once.
  Status Initialize();
  // Adds a new worker to the cluster.
  Status AddWorker();
  // Returns the dispatcher address in the form "hostname:port".
  std::string DispatcherAddress();
  // Returns the address of the worker at the specified index, in the form
  // "hostname:port". The index must be non-negative and less than the number of
  // workers in the cluster.
  std::string WorkerAddress(int index);

 private:
  bool initialized_ = false;
  int num_workers_;
  std::unique_ptr<DispatchGrpcDataServer> dispatcher_;
  std::string dispatcher_address_;
  std::vector<std::unique_ptr<WorkerGrpcDataServer>> workers_;
  std::vector<std::string> worker_addresses_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_TEST_CLUSTER_H_
