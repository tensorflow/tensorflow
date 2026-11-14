/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/checkpoint_reader.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_slice.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/io/table_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/tensor_bundle.pb.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace checkpoint {
namespace {

std::string TestPrefix(const std::string& name) {
  return absl::StrCat(testing::TmpDir(), "/", name);
}

absl::Status WriteCustomMetadataTable(
    const std::string& prefix,
    const std::function<void(table::TableBuilder*)>& populate) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(MetaFilename(prefix), &file));
  table::TableBuilder builder(table::Options(), file.get());

  BundleHeaderProto header;
  header.set_num_shards(1);
  header.set_endianness(BundleHeaderProto::LITTLE);
  header.mutable_version()->set_producer(kTensorBundleVersion);
  header.mutable_version()->set_min_consumer(kTensorBundleMinConsumer);
  builder.Add(kHeaderEntryKey, header.SerializeAsString());

  populate(&builder);
  TF_RETURN_IF_ERROR(builder.Finish());
  return file->Close();
}

TEST(CheckpointReaderTest, ValidV2Checkpoint) {
  const std::string prefix = TestPrefix("valid_v2");
  {
    BundleWriter writer(Env::Default(), prefix);
    Tensor t(DT_FLOAT, TensorShape({2, 3}));
    t.flat<float>().setConstant(1.0f);
    TF_ASSERT_OK(writer.Add("w", t));
    TF_ASSERT_OK(writer.Finish());
  }

  TF_Status* status = TF_NewStatus();
  CheckpointReader reader(prefix, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  EXPECT_EQ(reader.GetVariableToShapeMap().size(), 1u);
  EXPECT_EQ(reader.GetVariableToShapeMap().at("w"), TensorShape({2, 3}));
  EXPECT_EQ(reader.GetVariableToDataTypeMap().at("w"), DT_FLOAT);
  TF_DeleteStatus(status);
}

TEST(CheckpointReaderTest, CorruptBundleEntryProtoReturnsDataLoss) {
  const std::string prefix = TestPrefix("corrupt_entry_proto");
  TF_ASSERT_OK(WriteCustomMetadataTable(prefix, [](table::TableBuilder* b) {
    b->Add("var1", "not_a_valid_bundle_entry_proto\xff\xff\xff");
  }));

  TF_Status* status = TF_NewStatus();
  CheckpointReader reader(prefix, status);
  EXPECT_EQ(TF_GetCode(status), TF_DATA_LOSS) << TF_Message(status);
  TF_DeleteStatus(status);
}

TEST(CheckpointReaderTest, InvalidTensorShapeInEntryReturnsDataLoss) {
  const std::string prefix = TestPrefix("invalid_shape_entry");
  TF_ASSERT_OK(WriteCustomMetadataTable(prefix, [](table::TableBuilder* b) {
    BundleEntryProto entry;
    entry.set_dtype(DT_FLOAT);
    entry.mutable_shape()->add_dim()->set_size(-5);
    b->Add("var_negative_dim", entry.SerializeAsString());
  }));

  TF_Status* status = TF_NewStatus();
  CheckpointReader reader(prefix, status);
  EXPECT_EQ(TF_GetCode(status), TF_DATA_LOSS) << TF_Message(status);
  TF_DeleteStatus(status);
}

TEST(CheckpointReaderTest, InvalidTensorSliceInEntryReturnsDataLoss) {
  const std::string prefix = TestPrefix("invalid_slice_entry");
  TF_ASSERT_OK(WriteCustomMetadataTable(prefix, [](table::TableBuilder* b) {
    BundleEntryProto entry;
    entry.set_dtype(DT_FLOAT);
    entry.mutable_shape()->add_dim()->set_size(10);
    auto* slice = entry.add_slices();
    auto* extent = slice->add_extent();
    extent->set_start(-1);
    extent->set_length(5);
    b->Add("partitioned_var", entry.SerializeAsString());
  }));

  TF_Status* status = TF_NewStatus();
  CheckpointReader reader(prefix, status);
  EXPECT_EQ(TF_GetCode(status), TF_DATA_LOSS) << TF_Message(status);
  TF_DeleteStatus(status);
}

TEST(CheckpointReaderTest, DuplicateTensorSliceInEntryReturnsDataLoss) {
  const std::string prefix = TestPrefix("duplicate_slice_entry");
  TF_ASSERT_OK(WriteCustomMetadataTable(prefix, [](table::TableBuilder* b) {
    BundleEntryProto entry;
    entry.set_dtype(DT_FLOAT);
    entry.mutable_shape()->add_dim()->set_size(10);
    for (int i = 0; i < 2; ++i) {
      auto* slice = entry.add_slices();
      auto* extent = slice->add_extent();
      extent->set_start(0);
      extent->set_length(5);
    }
    b->Add("partitioned_var", entry.SerializeAsString());
  }));

  TF_Status* status = TF_NewStatus();
  CheckpointReader reader(prefix, status);
  EXPECT_EQ(TF_GetCode(status), TF_DATA_LOSS) << TF_Message(status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace checkpoint
}  // namespace tensorflow
