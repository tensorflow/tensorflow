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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_MEASURING_COST_ESTIMATOR_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_MEASURING_COST_ESTIMATOR_H_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class CostGraphDef;
class GraphDef;
}  // namespace tensorflow

namespace tensorflow {
namespace grappler {

class Cluster;
struct GrapplerItem;

// Estimate the cost of running a Grappler item by actually running the
// corresponding TensorFlow graph on the specified cluster and measuring the
// runtimes.
class MeasuringCostEstimator : public CostEstimator {
 public:
  // Run the model for measurement_steps to measure its average cost.
  // When measurement_threads is greater than 0, use a threadpool of as many
  // threads to run the measurements; otherwise, run them serially. Does not
  // take ownership of cluster.
  explicit MeasuringCostEstimator(Cluster* cluster, int measurement_steps,
                                  int measurement_threads);
  ~MeasuringCostEstimator() override {}

  // Initializes the estimator for the specified grappler item.
  // This implementation always returns OK.
  Status Initialize(const GrapplerItem& item) override;

  // Runs the optimized version of the graph on the cluster, measures
  // the runtimes of each operation, and annotates the CostGraphDef of
  // RunMetadata with the corresponding measurements.
  // Returns the average latency for the whole graph.
  Status PredictCosts(const GraphDef& optimized_graph,
                      RunMetadata* run_metadata, Costs* cost) const override;

 private:
  Cluster* cluster_;  // Not owned.
  int measurement_steps_;
  int measurement_threads_;
  std::vector<std::pair<string, Tensor>> feed_;
  std::vector<string> fetch_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_MEASURING_COST_ESTIMATOR_H_
