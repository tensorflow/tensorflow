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

#include "tensorflow/core/platform/cloud/compute_engine_metadata_client.h"

#include <utility>
#include "tensorflow/core/platform/cloud/curl_http_request.h"
#include "tensorflow/core/platform/cloud/retrying_utils.h"

namespace tensorflow {

namespace {

// The URL to retrieve metadata when running in Google Compute Engine.
constexpr char kGceMetadataBaseUrl[] = "http://metadata/computeMetadata/v1/";
// The default initial delay between retries with exponential backoff.
constexpr int kInitialRetryDelayUsec = 500000;  // 0.5 sec

}  // namespace

ComputeEngineMetadataClient::ComputeEngineMetadataClient(
    std::shared_ptr<HttpRequest::Factory> http_request_factory)
    : ComputeEngineMetadataClient(std::move(http_request_factory),
                                  kInitialRetryDelayUsec) {}

ComputeEngineMetadataClient::ComputeEngineMetadataClient(
    std::shared_ptr<HttpRequest::Factory> http_request_factory,
    int64 initial_retry_delay_usec)
    : http_request_factory_(std::move(http_request_factory)),
      initial_retry_delay_usec_(initial_retry_delay_usec) {}

Status ComputeEngineMetadataClient::GetMetadata(
    const string& path, std::vector<char>* response_buffer) {
  const auto get_metadata_from_gce = [path, response_buffer, this]() {
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    request->SetUri(kGceMetadataBaseUrl + path);
    request->AddHeader("Metadata-Flavor", "Google");
    request->SetResultBuffer(response_buffer);
    TF_RETURN_IF_ERROR(request->Send());
    return Status::OK();
  };

  return RetryingUtils::CallWithRetries(get_metadata_from_gce,
                                        initial_retry_delay_usec_);
}

}  // namespace tensorflow
