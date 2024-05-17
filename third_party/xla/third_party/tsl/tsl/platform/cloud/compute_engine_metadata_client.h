/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_CLOUD_COMPUTE_ENGINE_METADATA_CLIENT_H_
#define TENSORFLOW_TSL_PLATFORM_CLOUD_COMPUTE_ENGINE_METADATA_CLIENT_H_

#include "tsl/platform/cloud/http_request.h"
#include "tsl/platform/retrying_utils.h"
#include "tsl/platform/status.h"

namespace tsl {

/// \brief A client that accesses to the metadata server running on GCE hosts.
///
/// Uses the provided HttpRequest::Factory to make requests to the local
/// metadata service
/// (https://cloud.google.com/compute/docs/storing-retrieving-metadata).
/// Retries on recoverable failures using exponential backoff with the initial
/// retry wait configurable via initial_retry_delay_usec.
class ComputeEngineMetadataClient {
 public:
  explicit ComputeEngineMetadataClient(
      std::shared_ptr<HttpRequest::Factory> http_request_factory,
      const RetryConfig& config = RetryConfig(
          10000,  /* init_delay_time_us = 1 ms */
          1000000 /* max_delay_time_us = 1 s */
          ));
  virtual ~ComputeEngineMetadataClient() {}

  /// \brief Get the metadata value for a given attribute of the metadata
  /// service.
  ///
  /// Given a metadata path relative
  /// to http://metadata.google.internal/computeMetadata/v1/,
  /// fills response_buffer with the metadata. Returns OK if the server returns
  /// the response for the given metadata path successfully.
  ///
  /// Example usage:
  /// To get the zone of an instance:
  ///   compute_engine_metadata_client.GetMetadata(
  ///       "instance/zone", response_buffer);
  virtual Status GetMetadata(const string& path,
                             std::vector<char>* response_buffer);

 private:
  std::shared_ptr<HttpRequest::Factory> http_request_factory_;
  const RetryConfig retry_config_;

  ComputeEngineMetadataClient(const ComputeEngineMetadataClient&) = delete;
  void operator=(const ComputeEngineMetadataClient&) = delete;
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_CLOUD_COMPUTE_ENGINE_METADATA_CLIENT_H_
