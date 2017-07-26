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

#ifndef TENSORFLOW_GRAPPLER_CLUSTERS_SINGLE_MACHINE_H_
#define TENSORFLOW_GRAPPLER_CLUSTERS_SINGLE_MACHINE_H_

#include "tensorflow/cc/training/coordinator.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

// Create a simple cluster that makes available to grappler a subset of the
// nodes available on a single local computer.
class SingleMachine : public Cluster {
 public:
  SingleMachine(int timeout_s, int num_cpu_cores, int num_gpus);
  ~SingleMachine() override;

  Status Provision() override;
  Status Shutdown() override;

  Status Initialize(const GrapplerItem& item) override;
  Status Run(const GraphDef& item,
             const std::vector<std::pair<string, Tensor>>& feed,
             const std::vector<string>& fetch, RunMetadata* metadata) override;

 private:
  Status RunWithTimeout(const std::vector<std::pair<string, Tensor>>& feed,
                        const std::vector<string>& fetch,
                        RunMetadata* run_metadata);
  Status RunWithTimeout(const std::vector<std::pair<string, Tensor>>& feed,
                        const std::vector<string>& fetch,
                        RunMetadata* run_metadata, int64 timeout_s);
  Status ResetSession();
  Status CloseSession(bool use_timeout);
  void MergeCosts(CostGraphDef* graph_costs, const CostGraphDef& init_costs,
                  const CostGraphDef& queue_costs);

  const int num_gpus_;
  std::unique_ptr<Session> session_;
  std::vector<QueueRunnerDef> queue_runner_defs_;
  string last_graph_id_;
  mutex last_graph_mu_;
  const GraphDef* last_graph_ GUARDED_BY(last_graph_mu_) = nullptr;
  std::vector<string> init_ops_;
  int64 expected_init_time_s_;
  std::unique_ptr<Coordinator> coordinator_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;

  RunMetadata init_metadata_;

  mutex close_mu_;
  bool closing_ GUARDED_BY(close_mu_);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_CLUSTERS_SINGLE_MACHINE_H_
