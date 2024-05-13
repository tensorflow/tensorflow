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

#include "tensorflow/core/data/service/dataset_store.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/utils.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace data {

FileSystemDatasetStore::FileSystemDatasetStore(const std::string& datasets_dir)
    : datasets_dir_(datasets_dir) {}

Status FileSystemDatasetStore::Put(const std::string& key,
                                   const DatasetDef& dataset) {
  std::string path_to_write = io::JoinPath(datasets_dir_, key);
  TF_RETURN_IF_ERROR(WriteDatasetDef(path_to_write, dataset));
  return absl::OkStatus();
}

Status FileSystemDatasetStore::Get(
    const std::string& key, std::shared_ptr<const DatasetDef>& dataset_def) {
  std::string path = io::JoinPath(datasets_dir_, key);
  TF_RETURN_IF_ERROR(Env::Default()->FileExists(path));
  DatasetDef def;
  TF_RETURN_IF_ERROR(ReadDatasetDef(path, def));
  dataset_def = std::make_shared<const DatasetDef>(def);
  return absl::OkStatus();
}

Status MemoryDatasetStore::Put(const std::string& key,
                               const DatasetDef& dataset) {
  auto& stored_dataset = datasets_[key];
  stored_dataset = std::make_shared<const DatasetDef>(dataset);
  return absl::OkStatus();
}

Status MemoryDatasetStore::Get(const std::string& key,
                               std::shared_ptr<const DatasetDef>& dataset_def) {
  auto& stored_dataset = datasets_[key];
  if (!stored_dataset) {
    return errors::NotFound("Dataset with key ", key, " not found");
  }
  dataset_def = stored_dataset;
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
