/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_AUTO_SCALER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_AUTO_SCALER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/time/time.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/status.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// Estimates the optimal number of tf.data service workers for an Iteration
// based on the current workload.
// Note: It is assumed that all reported times correspond to the same Iteration.
//
// Glossary:
// * Consumer: A client that consumes elements from tf.data service.
// * Worker: A tf.data service worker.
// * Processing time (PT): The estimated time it takes a worker to process and
// produce an element.
// * Target processing time (TPT): From the perspective of a consumer,
// it is the maximum time a tf.data input pipeline can take to produce an
// element such that the downstream processor wait time is 0. In other words,
// this is the ideal time the tf.data pipeline should take to produce an element
// so that training doesn't slow down due to waiting for elements. This means
// that we want processing time <= target processing time, so that when an
// element is requested, the pipeline has processed it already.
// * Worker throughput (WT): It is the multiplicative inverse of processing time
// (1 / PT). This refers to the number of elements produced by a worker per
// second.
// * Consumption rate (CR): It is the multiplicative inverse of target
// processing time (1 / TPT). This refers to the number of elements requested by
// a consumer per second.
//
// **AutoScaler overview**
//
// 1. It keeps track of the most recent worker throughputs reported by each
// worker in the data service cluster, as well as the most recent consumption
// rates reported by each consumer. WTs and CRs are derived from reporting PTs
// and TPTs, respectively.
// 2. Having this information, it estimates the optimal number of workers N as
// follows:
//  N = (Sum of CRs reported by all consumers) /
//      (Average of WTs reported by all workers)
//
// AutoScaler is thread-safe.
class AutoScaler {
 public:
  AutoScaler() = default;
  // Returns the estimated optimal number of workers according to the current
  // observed workload. If there are no previously reported processing and
  // target processing times, returns nullopt.
  std::optional<int64_t> GetOptimalNumberOfWorkers() const
      TF_LOCKS_EXCLUDED(mu_);
  // Reports the latest observed processing time from the worker with
  // `worker_address`. Returns an error if `processing_time` is ZeroDuration or
  // negative.
  absl::Status ReportProcessingTime(const std::string& worker_address,
                                    absl::Duration processing_time)
      TF_LOCKS_EXCLUDED(mu_);
  // Reports the latest observed target processing time from the consumer
  // identified by `consumer_id`. Returns an error if `target_processing_time`
  // is ZeroDuration or negative.
  absl::Status ReportTargetProcessingTime(int64_t consumer_id,
                                          absl::Duration target_processing_time)
      TF_LOCKS_EXCLUDED(mu_);
  // Unregisters the worker with `worker_address`, removing its reported
  // processing time from consideration of the current workload estimation.
  // Returns an error if the specified worker does not exist.
  absl::Status RemoveWorker(const std::string& worker_address)
      TF_LOCKS_EXCLUDED(mu_);
  // Unregisters the consumer identified by `consumer_id`, removing its reported
  // target processing time from consideration of the current workload
  // estimation. Returns an error if the specified consumer does not exist.
  absl::Status RemoveConsumer(int64_t consumer_id) TF_LOCKS_EXCLUDED(mu_);

 private:
  mutable tsl::mutex mu_;
  // Map from worker address to worker throughput.
  absl::flat_hash_map<std::string, double> worker_throughputs_
      TF_GUARDED_BY(mu_);
  // Map from consumer id to consumption rate.
  absl::flat_hash_map<int64_t, double> consumption_rates_ TF_GUARDED_BY(mu_);
};

// Exports a metric (/tensorflow/data/service/optimal_number_of_workers) with
// the estimated optimal number of tf.data service workers, according to
// the observed cluster workload.
//
// It estimates the number of workers as the maximum of the estimated optimal
// number of workers for all Iterations running in the tf.data service cluster.
//
// MultipleIterationsAutoScaler is thread-safe.
class MultipleIterationsAutoScaler {
 public:
  MultipleIterationsAutoScaler() = default;
  // Unregisters iteration with `iteration_id`, removing its reported
  // times from consideration of the current workload estimation.
  // Returns an error if the specified iteration does not exist.
  absl::Status UnregisterIteration(int64_t iteration_id) TF_LOCKS_EXCLUDED(mu_);
  // Updates the metric value with the current estimated optimal number of
  // workers. The estimate is limited to min(4 * `current_number_of_workers`,
  // `current_number_of_workers` + 500). Returns an error if there are no
  // previously reported processing and target processing times for at least one
  // iteration, or `current_number_of_workers` is not positive.
  absl::Status UpdateOptimalNumberOfWorkersMetric(
      int64_t current_number_of_workers) TF_LOCKS_EXCLUDED(mu_);
  // Returns the estimated optimal number of workers according to the current
  // observed workload. If there are no previously reported processing and
  // target processing times for at least one iteration, returns nullopt.
  std::optional<int64_t> GetOptimalNumberOfWorkers() const
      TF_LOCKS_EXCLUDED(mu_);
  // Reports the latest observed processing time from the worker with
  // `worker_address` for iteration with `iteration_id`. Returns an error if
  // `processing_time` is ZeroDuration or negative.
  absl::Status ReportProcessingTime(int64_t iteration_id,
                                    const std::string& worker_address,
                                    absl::Duration processing_time)
      TF_LOCKS_EXCLUDED(mu_);
  // Reports the latest observed target processing time from the consumer
  // identified by `consumer_id` for iteration with `iteration_id`. Returns an
  // error if `target_processing_time` is ZeroDuration or negative.
  absl::Status ReportTargetProcessingTime(int64_t iteration_id,
                                          int64_t consumer_id,
                                          absl::Duration target_processing_time)
      TF_LOCKS_EXCLUDED(mu_);
  // Unregisters the worker with `worker_address` for iteration with
  // `iteration_id`, removing its reported processing time from consideration of
  // the current workload estimation. Returns an error if there are no
  // previously reported processing times for iteration with `iteration_id` and
  // the specified worker.
  absl::Status RemoveWorker(int64_t iteration_id,
                            const std::string& worker_address)
      TF_LOCKS_EXCLUDED(mu_);
  // Unregisters the consumer identified by `consumer_id` for iteration with
  // `iteration_id`, removing its reported target processing time from
  // consideration of the current workload estimation. Returns an error if there
  // are no previously reported processing times for iteration with
  // `iteration_id` and the specified consumer.
  absl::Status RemoveConsumer(int64_t iteration_id, int64_t consumer_id)
      TF_LOCKS_EXCLUDED(mu_);

 private:
  // Registers iteration with `iteration_id` if it does not exist already,
  // allowing its future reported times to be considered for the current
  // workload estimation.
  void EnsureIterationIsRegistered(int64_t iteration_id)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  mutable tsl::mutex mu_;
  // Map from iteration id to AutoScaler.
  absl::flat_hash_map<int64_t, std::unique_ptr<AutoScaler>> auto_scalers_
      TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_AUTO_SCALER_H_
