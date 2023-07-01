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

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

tsl::Status AutoScaler::ReportProcessingTime(const std::string& worker_address,
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

  return tsl::OkStatus();
}

tsl::Status AutoScaler::ReportTargetProcessingTime(
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

  return tsl::OkStatus();
}

tsl::Status AutoScaler::RemoveWorker(const std::string& worker_address)
    TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  if (!worker_throughputs_.contains(worker_address))
    return absl::NotFoundError(
        absl::StrCat("Worker with address ", worker_address, " not found"));

  worker_throughputs_.erase(worker_address);

  return tsl::OkStatus();
}

tsl::Status AutoScaler::RemoveConsumer(int64_t consumer_id)
    TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  if (!consumption_rates_.contains(consumer_id))
    return absl::NotFoundError(
        absl::StrCat("Consumer with ID ", consumer_id, " not found"));

  consumption_rates_.erase(consumer_id);

  return tsl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
