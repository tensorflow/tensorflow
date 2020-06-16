/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <stdlib.h>
#include <string.h>

#include "google/cloud/storage/client.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for GCS environments.
// This filesystem will support `gs://` URI schemes.
namespace gcs = google::cloud::storage;

// We can cast `google::cloud::StatusCode` to `TF_Code` because they have the
// same integer values. See
// https://github.com/googleapis/google-cloud-cpp/blob/6c09cbfa0160bc046e5509b4dd2ab4b872648b4a/google/cloud/status.h#L32-L52
static inline void TF_SetStatusFromGCSStatus(
    const google::cloud::Status& gcs_status, TF_Status* status) {
  TF_SetStatus(status, static_cast<TF_Code>(gcs_status.code()),
               gcs_status.message().c_str());
}

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) { free(ptr); }

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

// TODO(vnvo2409): Implement later

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

// TODO(vnvo2409): Implement later

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

// TODO(vnvo2409): Implement later

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_gcs_filesystem {

// TODO(vnvo2409): Add lazy-loading and customizing parameters.
static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  google::cloud::StatusOr<gcs::Client> client =
      gcs::Client::CreateDefaultClient();
  if (!client) {
    TF_SetStatusFromGCSStatus(client.status(), status);
    return;
  }
  filesystem->plugin_filesystem = plugin_memory_allocate(sizeof(gcs::Client));
  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  (*gcs_client) = client.value();
  TF_SetStatus(status, TF_OK, "");
}

// TODO(vnvo2409): Implement later

}  // namespace tf_gcs_filesystem

static void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops,
                                        const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
      plugin_memory_allocate(TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_gcs_filesystem::Init;
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = plugin_memory_allocate;
  info->plugin_memory_free = plugin_memory_free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      plugin_memory_allocate(info->num_schemes * sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "gs");
}
