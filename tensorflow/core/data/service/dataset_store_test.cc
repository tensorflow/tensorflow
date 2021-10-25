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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

namespace {
const char kFileSystem[] = "file_system";
const char kMemory[] = "memory";

std::string NewDatasetsDir() {
  std::string dir = io::JoinPath(testing::TmpDir(), "datasets");
  if (Env::Default()->FileExists(dir).ok()) {
    int64_t undeleted_files;
    int64_t undeleted_dirs;
    CHECK(Env::Default()
              ->DeleteRecursively(dir, &undeleted_files, &undeleted_dirs)
              .ok());
  }
  CHECK(Env::Default()->RecursivelyCreateDir(dir).ok());
  return dir;
}

std::unique_ptr<DatasetStore> MakeStore(const std::string& type) {
  if (type == kFileSystem) {
    return absl::make_unique<FileSystemDatasetStore>(NewDatasetsDir());
  } else if (type == kMemory) {
    return absl::make_unique<MemoryDatasetStore>();
  } else {
    CHECK(false) << "unexpected type: " << type;
  }
}

DatasetDef DatasetDefWithVersion(int32_t version) {
  DatasetDef def;
  def.mutable_graph()->set_version(version);
  return def;
}

}  // namespace

class DatasetStoreTest : public ::testing::Test,
                         public ::testing::WithParamInterface<std::string> {};

TEST_P(DatasetStoreTest, StoreAndGet) {
  std::unique_ptr<DatasetStore> store = MakeStore(GetParam());
  std::string key = "key";
  DatasetDef dataset_def = DatasetDefWithVersion(1);
  TF_ASSERT_OK(store->Put(key, dataset_def));
  std::shared_ptr<const DatasetDef> result;
  TF_ASSERT_OK(store->Get(key, result));
  EXPECT_EQ(result->graph().version(), dataset_def.graph().version());
}

TEST_P(DatasetStoreTest, StoreAndGetMultiple) {
  std::unique_ptr<DatasetStore> store = MakeStore(GetParam());
  int64_t num_datasets = 10;
  std::vector<std::string> keys;
  for (int i = 0; i < num_datasets; ++i) {
    std::string key = absl::StrCat("key", i);
    DatasetDef dataset_def = DatasetDefWithVersion(i);
    TF_ASSERT_OK(store->Put(key, dataset_def));
    keys.push_back(key);
  }
  for (int i = 0; i < num_datasets; ++i) {
    std::shared_ptr<const DatasetDef> result;
    TF_ASSERT_OK(store->Get(keys[i], result));
    EXPECT_EQ(result->graph().version(), i);
  }
}

TEST_P(DatasetStoreTest, StoreAlreadyExists) {
  std::unique_ptr<DatasetStore> store = MakeStore(GetParam());
  int32_t version = 1;
  DatasetDef dataset_def = DatasetDefWithVersion(version);
  std::string key = "key";
  TF_ASSERT_OK(store->Put(key, dataset_def));
  Status s = store->Put(key, dataset_def);
  EXPECT_EQ(s.code(), error::ALREADY_EXISTS);
  std::shared_ptr<const DatasetDef> result;
  TF_ASSERT_OK(store->Get(key, result));
  EXPECT_EQ(result->graph().version(), version);
}

TEST_P(DatasetStoreTest, GetMissing) {
  std::unique_ptr<DatasetStore> store = MakeStore(GetParam());
  std::shared_ptr<const DatasetDef> result;
  Status s = store->Get("missing", result);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

INSTANTIATE_TEST_SUITE_P(DatasetStoreTests, DatasetStoreTest,
                         ::testing::Values(kFileSystem, kMemory));
}  // namespace data
}  // namespace tensorflow
