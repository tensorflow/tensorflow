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

#include "tensorflow/core/platform/cloud/gcs_throttle.h"

#include <algorithm>

namespace tensorflow {

GcsThrottle::GcsThrottle(EnvTime* env_time)
    : last_updated_secs_(env_time->NowSeconds()),
      available_tokens_(0),
      env_time_(env_time) {}

bool GcsThrottle::AdmitRequest() {
  mutex_lock l(mu_);
  UpdateState();
  if (available_tokens_ < config_.tokens_per_request) {
    return false || !config_.enabled;
  }
  available_tokens_ -= config_.tokens_per_request;
  return true;
}

void GcsThrottle::RecordResponse(size_t num_bytes) {
  mutex_lock l(mu_);
  UpdateState();
  available_tokens_ -= request_bytes_to_tokens(num_bytes);
}

void GcsThrottle::SetConfig(GcsThrottleConfig config) {
  mutex_lock l(mu_);
  config_ = config;
  available_tokens_ = config.initial_tokens;
  last_updated_secs_ = env_time_->NowSeconds();
}

void GcsThrottle::UpdateState() {
  // TODO(b/72643279): Switch to a monotonic clock.
  int64 now = env_time_->NowSeconds();
  uint64 delta_secs =
      std::max(int64{0}, now - static_cast<int64>(last_updated_secs_));
  available_tokens_ += delta_secs * config_.token_rate;
  available_tokens_ = std::min(available_tokens_, config_.bucket_size);
  last_updated_secs_ = now;
}

}  // namespace tensorflow
