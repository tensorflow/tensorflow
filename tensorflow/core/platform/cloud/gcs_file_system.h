/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_FILE_SYSTEM_H_

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/cloud/auth_provider.h"
#include "tensorflow/core/platform/cloud/compute_engine_metadata_client.h"
#include "tensorflow/core/platform/cloud/compute_engine_zone_provider.h"
#include "tensorflow/core/platform/cloud/expiring_lru_cache.h"
#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include "tensorflow/core/platform/cloud/gcs_dns_cache.h"
#include "tensorflow/core/platform/cloud/gcs_throttle.h"
#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/retrying_file_system.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/cloud/gcs_file_system.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::GcsFileSystem;
using tsl::GcsStatsInterface;
using tsl::GetEnvVar;
using tsl::kBlockSize;
using tsl::kDefaultBlockSize;
using tsl::kDefaultMaxCacheSize;
using tsl::kDefaultMaxStaleness;
using tsl::kMaxCacheSize;
using tsl::kMaxStaleness;
using tsl::RetryingGcsFileSystem;
using tsl::UploadSessionHandle;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_FILE_SYSTEM_H_
