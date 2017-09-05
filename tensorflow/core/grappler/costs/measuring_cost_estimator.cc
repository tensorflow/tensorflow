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

#include "tensorflow/core/grappler/costs/measuring_cost_estimator.h"

#include <limits>

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/robust_stats.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

MeasuringCostEstimator::MeasuringCostEstimator(Cluster* cluster,
                                               int measurement_steps,
                                               int measurement_threads)
    : measurement_steps_(measurement_steps),
      measurement_threads_(measurement_threads) {
  CHECK_GE(measurement_steps, 1);
  if (measurement_threads > 0) {
    thread_pool_.reset(new thread::ThreadPool(
        Env::Default(), SanitizeThreadSuffix("measurements"),
        measurement_threads));
  }
  cluster_ = cluster;
}

Status MeasuringCostEstimator::Initialize(const GrapplerItem& item) {
  feed_ = item.feed;
  fetch_ = item.fetch;
  return cluster_->Initialize(item);
}

Status MeasuringCostEstimator::PredictCosts(const GraphDef& optimized_graph,
                                            CostGraphDef* cost_graph,
                                            Costs* costs) const {
  std::vector<double> times(measurement_steps_);
  BlockingCounter barrier(measurement_steps_);

  mutex status_mu;
  Status status;

  auto measurement_fn = [&](const int step) {
    const Costs::MicroSeconds start = Env::Default()->NowMicros();

    RunMetadata metadata;
    const Status local_status =
        cluster_->Run(optimized_graph, feed_, fetch_, &metadata);
    {
      mutex_lock lock(status_mu);
      status.Update(local_status);
    }
    if (step < 0) {
      // Discard the first iteration as it triggers the warmup, and therefore
      // takes much longer than a normal step.
      return;
    }
    if (!local_status.ok()) {
      // Discard the data if the run wasn't successful.
      barrier.DecrementCount();
      return;
    }

    const Costs::MicroSeconds finish = Env::Default()->NowMicros();
    const double time = (finish - start).count() * 1e3;
    times[step] = time;

    if (cost_graph && (step + 1 == measurement_steps_)) {
      metadata.mutable_cost_graph()->Swap(cost_graph);
    }

    barrier.DecrementCount();
  };

  // Initialize the computation and warm up TensorFlow.
  measurement_fn(-1);

  if (!status.ok()) {
    LOG(ERROR) << "Failed to run start measurements: "
               << status.error_message();
    costs->execution_time = Costs::Duration::max();
    return status;
  }

  // Run "measurement_steps_" and measure the time.
  VLOG(1) << "Number of measurement steps: " << measurement_steps_;
  if (measurement_threads_ > 0) {
    for (int i = 0; i < measurement_steps_; ++i) {
      thread_pool_->Schedule([i, &measurement_fn]() { measurement_fn(i); });
    }
    barrier.Wait();
  } else {
    for (int i = 0; i < measurement_steps_ && status.ok(); ++i) {
      measurement_fn(i);
    }
  }

  if (!status.ok()) {
    LOG(ERROR) << "Failed to measure graph performance: "
               << status.error_message();
    costs->execution_time = Costs::Duration::max();
    costs->max_execution_time = Costs::Duration::max();
    costs->min_execution_time = 0;
    return status;
  }

  // Compute the average time of the measure steps. Use Huber statistics
  // to filter out outliers.
  RobustStats stats(times);
  costs->execution_time = Costs::Duration(stats.mean());
  costs->max_execution_time = Costs::Duration(stats.hi());
  costs->min_execution_time = Costs::Duration(stats.lo());

  return Status::OK();
}
}  // end namespace grappler
}  // end namespace tensorflow
