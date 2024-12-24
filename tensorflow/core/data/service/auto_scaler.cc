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

#include "tensorflow/core/data/service/auto_scaler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/core/framework/metrics.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

constexpr double kAutoScalerOutlierSigmas = 1.0;

template <typename T>
double GetMedian(const absl::flat_hash_map<T, double>& rates) {
  std::vector<double> sorted_rates;
  for (const auto& [id, rate] : rates) {
    sorted_rates.push_back(rate);
  }
  std::sort(sorted_rates.begin(), sorted_rates.end());

  return sorted_rates[sorted_rates.size() / 2];
}

template <typename T>
double GetMean(const absl::flat_hash_map<T, double>& rates) {
  double rates_sum = 0.0;
  for (const auto& [id, rate] : rates) {
    rates_sum += rate;
  }
  if (rates_sum == 0.0) return 0.0;

  return rates_sum / static_cast<double>(rates.size());
}

template <typename T>
double GetStandardDeviation(const absl::flat_hash_map<T, double>& rates,
                            double mean) {
  double squared_distances_sum = 0.0;
  for (const auto& [id, rate] : rates) {
    squared_distances_sum += (rate - mean) * (rate - mean);
  }
  if (squared_distances_sum == 0.0 || rates.size() <= 1) return 0.0;

  return std::sqrt(squared_distances_sum /
                   static_cast<double>(rates.size() - 1));
}

// Discards rates that are more than (std_dev * outlier_sigmas) far from the
// mean, and replaces them with the median. Puts the result in
// `rates_without_outliers`.
template <typename T>
void ReplaceOutliers(const absl::flat_hash_map<T, double>& rates,
                     std::vector<double>& rates_without_outliers,
                     double outlier_sigmas) {
  if (rates.empty()) return;
  double mean = GetMean(rates);
  double median = GetMedian(rates);
  double standard_deviation = GetStandardDeviation(rates, mean);

  double lower_threshold = mean - standard_deviation * outlier_sigmas;
  double upper_threshold = mean + standard_deviation * outlier_sigmas;

  for (const auto& [id, rate] : rates) {
    if (rate >= lower_threshold && rate <= upper_threshold) {
      rates_without_outliers.push_back(rate);
    } else {
      rates_without_outliers.push_back(median);
    }
  }
}

std::optional<int64_t> AutoScaler::GetOptimalNumberOfWorkers() const
    TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);

  if (worker_throughputs_.empty() || consumption_rates_.empty())
    return std::nullopt;

  std::vector<double> consumption_rates_without_outliers;
  // TODO(armandouv): Discard outlier replacement when we ensure reported time
  // values are correct.
  // Outliers can make the estimate have an unfeasible value (very high or very
  // low).
  ReplaceOutliers(consumption_rates_, consumption_rates_without_outliers,
                  kAutoScalerOutlierSigmas);
  double consumption_rates_sum_ =
      std::accumulate(consumption_rates_without_outliers.begin(),
                      consumption_rates_without_outliers.end(), 0.0);

  std::vector<double> worker_throughputs_without_outliers;
  ReplaceOutliers(worker_throughputs_, worker_throughputs_without_outliers,
                  kAutoScalerOutlierSigmas);
  double worker_throughputs_sum_ =
      std::accumulate(worker_throughputs_without_outliers.begin(),
                      worker_throughputs_without_outliers.end(), 0.0);

  double average_worker_throughput =
      worker_throughputs_sum_ / static_cast<double>(worker_throughputs_.size());

  int64_t optimal_number_of_workers =
      ceil(consumption_rates_sum_ / average_worker_throughput);

  return std::max(int64_t{1}, optimal_number_of_workers);
}

absl::Status AutoScaler::ReportProcessingTime(const std::string& worker_address,
                                              absl::Duration processing_time)
    TF_LOCKS_EXCLUDED(mu_) {
  if (processing_time <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot update processing_time with a ZeroDuration or negative value: ",
        absl::FormatDuration(processing_time)));
  }

  double worker_throughput = 1.0 / absl::ToDoubleSeconds(processing_time);
  tsl::mutex_lock l(mu_);
  worker_throughputs_[worker_address] = worker_throughput;

  return absl::OkStatus();
}

absl::Status AutoScaler::ReportTargetProcessingTime(
    int64_t consumer_id, absl::Duration target_processing_time)
    TF_LOCKS_EXCLUDED(mu_) {
  if (target_processing_time <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot update target_processing_time with a ZeroDuration "
                     "or negative value: ",
                     absl::FormatDuration(target_processing_time)));
  }

  double consumption_rate = 1.0 / absl::ToDoubleSeconds(target_processing_time);
  tsl::mutex_lock l(mu_);
  consumption_rates_[consumer_id] = consumption_rate;

  return absl::OkStatus();
}

absl::Status AutoScaler::RemoveWorker(const std::string& worker_address)
    TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  if (!worker_throughputs_.contains(worker_address))
    return absl::NotFoundError(
        absl::StrCat("Worker with address ", worker_address, " not found"));

  worker_throughputs_.erase(worker_address);

  return absl::OkStatus();
}

absl::Status AutoScaler::RemoveConsumer(int64_t consumer_id)
    TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  if (!consumption_rates_.contains(consumer_id))
    return absl::NotFoundError(
        absl::StrCat("Consumer with ID ", consumer_id, " not found"));

  consumption_rates_.erase(consumer_id);

  return absl::OkStatus();
}

void MultipleIterationsAutoScaler::EnsureIterationIsRegistered(
    int64_t iteration_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (!auto_scalers_.contains(iteration_id)) {
    auto_scalers_[iteration_id] = std::make_unique<AutoScaler>();
  }
}

absl::Status MultipleIterationsAutoScaler::UnregisterIteration(
    int64_t iteration_id) TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  if (!auto_scalers_.contains(iteration_id))
    return absl::NotFoundError(absl::StrCat("AutoScaler for iteration_id ",
                                            iteration_id, " does not exist"));
  auto_scalers_.erase(iteration_id);
  return absl::OkStatus();
}

absl::Status MultipleIterationsAutoScaler::UpdateOptimalNumberOfWorkersMetric(
    int64_t current_number_of_workers) TF_LOCKS_EXCLUDED(mu_) {
  if (current_number_of_workers <= 0)
    return absl::InvalidArgumentError(
        "The current number of workers must be positive");

  std::optional<int64_t> optimal_number_of_workers =
      GetOptimalNumberOfWorkers();
  if (!optimal_number_of_workers)
    return absl::UnavailableError(
        "Cannot update the optimal number of workers metric because there are "
        "no reported processing and target processing times for at least one "
        "iteration");

  VLOG(3) << "Estimated optimal number of workers: "
          << optimal_number_of_workers.value();

  // Limit the estimate to wait for target processing times to converge to a
  // feasible value. First, start increasing exponentially by 4x. Once
  // increases are greater than 500, scale linearly.
  int64_t bound_optimal_number_of_workers = optimal_number_of_workers.value();
  if (bound_optimal_number_of_workers > current_number_of_workers * 4 ||
      bound_optimal_number_of_workers > current_number_of_workers + 500) {
    bound_optimal_number_of_workers = std::min(current_number_of_workers * 4,
                                               current_number_of_workers + 500);
  }
  // Limit the estimate to at most 100k workers.
  bound_optimal_number_of_workers =
      std::min(bound_optimal_number_of_workers, int64_t{100000});
  VLOG(3) << "Bound optimal number of workers: "
          << bound_optimal_number_of_workers;

  metrics::RecordTFDataServiceOptimalNumberOfWorkers(
      bound_optimal_number_of_workers);

  return absl::OkStatus();
}

std::optional<int64_t> MultipleIterationsAutoScaler::GetOptimalNumberOfWorkers()
    const TF_LOCKS_EXCLUDED(mu_) {
  int64_t optimal_number_of_workers = 0;
  {
    tsl::tf_shared_lock l(mu_);
    for (const auto& [iteration_id, auto_scaler] : auto_scalers_) {
      std::optional<int64_t> current_optimal_number_of_workers =
          auto_scaler->GetOptimalNumberOfWorkers();
      if (!current_optimal_number_of_workers.has_value()) continue;

      optimal_number_of_workers = std::max(
          optimal_number_of_workers, current_optimal_number_of_workers.value());
    }
  }

  if (optimal_number_of_workers == 0)
    return std::nullopt;
  else
    return optimal_number_of_workers;
}

absl::Status MultipleIterationsAutoScaler::ReportProcessingTime(
    int64_t iteration_id, const std::string& worker_address,
    absl::Duration processing_time) TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  EnsureIterationIsRegistered(iteration_id);

  absl::Status status = auto_scalers_[iteration_id]->ReportProcessingTime(
      worker_address, processing_time);
  return status;
}

absl::Status MultipleIterationsAutoScaler::ReportTargetProcessingTime(
    int64_t iteration_id, int64_t consumer_id,
    absl::Duration target_processing_time) TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  EnsureIterationIsRegistered(iteration_id);

  absl::Status status = auto_scalers_[iteration_id]->ReportTargetProcessingTime(
      consumer_id, target_processing_time);
  return status;
}

absl::Status MultipleIterationsAutoScaler::RemoveWorker(
    int64_t iteration_id, const std::string& worker_address)
    TF_LOCKS_EXCLUDED(mu_) {
  tsl::tf_shared_lock l(mu_);
  if (!auto_scalers_.contains(iteration_id))
    return absl::NotFoundError(absl::StrCat(
        "There are no reported times for iteration_id ", iteration_id));

  absl::Status status =
      auto_scalers_[iteration_id]->RemoveWorker(worker_address);
  return status;
}

absl::Status MultipleIterationsAutoScaler::RemoveConsumer(int64_t iteration_id,
                                                          int64_t consumer_id)
    TF_LOCKS_EXCLUDED(mu_) {
  tsl::tf_shared_lock l(mu_);
  if (!auto_scalers_.contains(iteration_id))
    return absl::NotFoundError(absl::StrCat(
        "There are no reported times for iteration_id ", iteration_id));

  absl::Status status =
      auto_scalers_[iteration_id]->RemoveConsumer(consumer_id);
  return status;
}

}  // namespace data
}  // namespace tensorflow
