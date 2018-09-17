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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_COMPUTE_ENGINE_ZONE_PROVIDER_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_COMPUTE_ENGINE_ZONE_PROVIDER_H_

#include "tensorflow/core/platform/cloud/compute_engine_metadata_client.h"
#include "tensorflow/core/platform/cloud/zone_provider.h"

namespace tensorflow {

class ComputeEngineZoneProvider : public ZoneProvider {
 public:
  explicit ComputeEngineZoneProvider(
      std::shared_ptr<ComputeEngineMetadataClient> google_metadata_client);
  virtual ~ComputeEngineZoneProvider();

  Status GetZone(string* zone) override;

 private:
  std::shared_ptr<ComputeEngineMetadataClient> google_metadata_client_;
  string cached_zone;
  TF_DISALLOW_COPY_AND_ASSIGN(ComputeEngineZoneProvider);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_COMPUTE_ENGINE_ZONE_PROVIDER_H_
