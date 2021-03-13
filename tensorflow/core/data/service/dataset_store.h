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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_DATASET_STORE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DATASET_STORE_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {

// An interface for storing and getting dataset definitions.
class DatasetStore {
 public:
  virtual ~DatasetStore() = default;

  // Stores the given dataset under the given key. Returns ALREADY_EXISTS if the
  // key already exists.
  virtual Status Put(const std::string& key, const DatasetDef& dataset) = 0;
  // Gets the dataset for the given key, storing the dataset in `dataset_def`.
  virtual Status Get(const std::string& key,
                     std::shared_ptr<const DatasetDef>& dataset_def) = 0;
};

// Dataset store which reads and writes datasets within a directory.
// The dataset with key `key` is stored at the path "datasets_dir/key".
class FileSystemDatasetStore : public DatasetStore {
 public:
  explicit FileSystemDatasetStore(const std::string& datasets_dir);
  FileSystemDatasetStore(const FileSystemDatasetStore&) = delete;
  FileSystemDatasetStore& operator=(const FileSystemDatasetStore&) = delete;

  Status Put(const std::string& key, const DatasetDef& dataset) override;
  Status Get(const std::string& key,
             std::shared_ptr<const DatasetDef>& dataset_def) override;

 private:
  const std::string datasets_dir_;
};

// DatasetStore which stores all datasets in memory. This is useful when the
// dispatcher doesn't have a work directory configured.
class MemoryDatasetStore : public DatasetStore {
 public:
  MemoryDatasetStore();
  MemoryDatasetStore(const MemoryDatasetStore&) = delete;
  MemoryDatasetStore& operator=(const MemoryDatasetStore&) = delete;

  Status Put(const std::string& key, const DatasetDef& dataset) override;
  Status Get(const std::string& key,
             std::shared_ptr<const DatasetDef>& dataset_def) override;

 private:
  // Mapping from key to dataset definition.
  absl::flat_hash_map<std::string, std::shared_ptr<const DatasetDef>> datasets_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DATASET_STORE_H_
