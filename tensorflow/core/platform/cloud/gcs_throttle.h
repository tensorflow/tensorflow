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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_THROTTLE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_THROTTLE_H_

#include "tensorflow/core/platform/env.h"

namespace tensorflow {

/**
 * GcsThrottleConfig is used to configure the GcsThrottle.
 */
struct GcsThrottleConfig {
  /**
   * enabled is true if GcsThrottle should throttle requests, false otherwise.
   */
  bool enabled = false;

  /**
   * token_rate is the number of tokens accrued every second that can be used
   * for making requests to the GCS service.
   */
  int64 token_rate = 100000;  // Approximately 800 MBits/second bandwidth-only.

  /**
   * bucket_size is the maximum number of available tokens the GcsThrottle can
   * accrue.
   */
  int64 bucket_size = 10000000;  // 10 million tokens total

  /**
   * tokens_per_request determines the number of tokens consumed for every
   * request.
   *
   * Note: tokens are also consumed in proportion to the response size.
   */
  int64 tokens_per_request = 100;

  /**
   * initial_tokens determines how many tokens should be available immediately
   * after the GcsThrottle is constructed.
   */
  int64 initial_tokens = 0;
};

/**
 * GcsThrottle is used to ensure fair use of the available GCS capacity.
 *
 * GcsThrottle operates around a concept of tokens. Tokens are consumed when
 * making requests to the GCS service. Tokens are consumed both based on the
 * number of requests made, as well as the bandwidth consumed (response sizes).
 *
 * GcsThrottle is thread safe and can be used from multiple threads.
 */
class GcsThrottle {
 public:
  /**
   * Constructs a GcsThrottle.
   */
  explicit GcsThrottle(EnvTime* env_time = EnvTime::Default());

  /**
   * AdmitRequest updates the GcsThrottle to record a request will be made.
   *
   * AdmitRequest should be called before any request is made. AdmitRequest
   * returns false if the request should be denied. If AdmitRequest
   * returns false, no tokens are consumed. If true is returned, the configured
   * number of tokens are consumed.
   */
  bool AdmitRequest();

  /**
   * RecordResponse updates the GcsThrottle to record a request has been made.
   *
   * RecordResponse should be called after the response has been received.
   * RecordResponse will update the internal state based on the number of bytes
   * in the response.
   *
   * Note: we split up the request and the response in this fashion in order to
   * avoid penalizing consumers who are using large readahead buffers at higher
   * layers of the I/O stack.
   */
  void RecordResponse(size_t num_bytes);

  /**
   * SetConfig sets the configuration for GcsThrottle and re-initializes state.
   *
   * After calling this, the token pool will be config.initial_tokens.
   */
  void SetConfig(GcsThrottleConfig config);

  /**
   * available_tokens gives a snapshot of how many tokens are available.
   *
   * The returned value should not be used to make admission decisions. The
   * purpose of this function is to make available to monitoring or other
   * instrumentation the number of available tokens in the pool.
   */
  inline int64 available_tokens() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    UpdateState();
    return available_tokens_;
  }

  /**
   * is_enabled determines if the throttle is enabled.
   *
   * If !is_enabled(), AdmitRequest() will always return true. To enable the
   * throttle, call SetConfig passing in a configuration that has enabled set to
   * true.
   */
  bool is_enabled() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    return config_.enabled;
  }

 private:
  /**
   * UpdateState updates the available_tokens_ and last_updated_secs_ variables.
   *
   * UpdateState should be called in order to mark the passage of time, and
   * therefore add tokens to the availble_tokens_ pool.
   */
  void UpdateState() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  inline uint64 request_bytes_to_tokens(size_t num_bytes) {
    return num_bytes >> 10;
  }

  mutex mu_;

  /**
   * last_updated_secs_ records the number of seconds since the Unix epoch that
   * the internal state of the GcsThrottle was updated. This is important when
   * determining the number of tokens to add to the available_tokens_ pool.
   */
  uint64 last_updated_secs_ GUARDED_BY(mu_) = 0;

  /**
   * available_tokens_ records how many tokens are available to be consumed.
   *
   * Note: it is possible for available_tokens_ to become negative. If a
   * response comes back that consumes more than the available tokens, the count
   * will go negative, and block future requests until we have available tokens.
   */
  int64 available_tokens_ GUARDED_BY(mu_) = 0;

  EnvTime* const env_time_;
  GcsThrottleConfig config_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_THROTTLE_H_
