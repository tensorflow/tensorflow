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
#include "tensorflow/core/data/service/utils.h"

#include <string>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {

namespace {
DatasetDef DatasetDefWithVersion(int32_t version) {
  DatasetDef def;
  def.mutable_graph()->set_version(version);
  return def;
}
}  // namespace

TEST(Utils, ReadWriteDataset) {
  std::string filename = testing::TmpDir();
  ASSERT_TRUE(Env::Default()->CreateUniqueFileName(&filename, "journal_dir"));
  int32_t version = 3;
  DatasetDef def = DatasetDefWithVersion(version);
  TF_ASSERT_OK(WriteDatasetDef(filename, def));
  DatasetDef result;
  TF_ASSERT_OK(ReadDatasetDef(filename, result));
  EXPECT_EQ(result.graph().version(), version);
}

TEST(Utils, OverwriteDataset) {
  std::string filename = testing::TmpDir();
  ASSERT_TRUE(Env::Default()->CreateUniqueFileName(&filename, "journal_dir"));
  int32_t version_1 = 1;
  int32_t version_2 = 2;
  DatasetDef def_1 = DatasetDefWithVersion(version_1);
  TF_ASSERT_OK(WriteDatasetDef(filename, def_1));
  DatasetDef def_2 = DatasetDefWithVersion(version_2);
  TF_ASSERT_OK(WriteDatasetDef(filename, def_2));
  DatasetDef result;
  TF_ASSERT_OK(ReadDatasetDef(filename, result));
  EXPECT_EQ(result.graph().version(), version_2);
}

TEST(Utils, ReadDatasetNotFound) {
  std::string filename = testing::TmpDir();
  ASSERT_TRUE(Env::Default()->CreateUniqueFileName(&filename, "journal_dir"));
  DatasetDef result;
  absl::Status s = ReadDatasetDef(filename, result);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

}  // namespace data
}  // namespace tensorflow
