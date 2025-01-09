/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <map>
#include <memory>
#include <string>

#include "fuzztest/fuzztest.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/io/table_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/util/saved_tensor_slice.pb.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/security/fuzzing/cc/checkpoint_reader_fuzz_input.pb.h"
#include "tsl/platform/status.h"

// This is a fuzzer for tensorflow::checkpoint::CheckpointReader. LevelDB
// reading and proto parsing are already fuzz-tested, so there's no need to test
// them here.

namespace {

using ::tensorflow::checkpoint::EncodeTensorNameSlice;
using ::tensorflow::checkpoint::kSavedTensorSlicesKey;

void CreateCheckpoint(
    const std::string& filename,
    const tensorflow::testing::CheckpointReaderFuzzInput& contents) {
  std::unique_ptr<tensorflow::WritableFile> writable_file;
  TF_CHECK_OK(
      tensorflow::Env::Default()->NewWritableFile(filename, &writable_file));
  tensorflow::table::Options options;
  options.compression = tensorflow::table::kNoCompression;
  tensorflow::table::TableBuilder builder(options, writable_file.get());

  // Entries must be added in sorted order.
  {
    tensorflow::SavedTensorSlices sts;
    *sts.mutable_meta() = contents.meta();
    builder.Add(kSavedTensorSlicesKey, sts.SerializeAsString());
  }
  std::map<std::string, const tensorflow::SavedSlice*> entries;
  for (const tensorflow::SavedSlice& saved_slice : contents.data()) {
    // The encoded tensor slice name is not included in the fuzz input since
    // it's difficult for the fuzzer to find the proper encoding, resulting in
    // lots of fruitless inputs with mismatched keys. Note that TensorSlice will
    // not currently crash with unverified data so long as it's only used by
    // EncodeTensorNameSlice.
    tensorflow::TensorSlice slice(saved_slice.slice());
    entries.insert(
        {EncodeTensorNameSlice(saved_slice.name(), slice), &saved_slice});
  }
  tensorflow::SavedTensorSlices sts;
  for (const auto& entry : entries) {
    *sts.mutable_data() = *entry.second;
    builder.Add(entry.first, sts.SerializeAsString());
  }
  TF_CHECK_OK(builder.Finish());
  TF_CHECK_OK(writable_file->Close());
}

int GetDataTypeSize(tensorflow::DataType data_type) {
  // tensorflow::DataTypeSize doesn't support several types.
  switch (data_type) {
    case tensorflow::DT_STRING:
      return sizeof(tensorflow::tstring);
    case tensorflow::DT_VARIANT:
      return sizeof(tensorflow::Variant);
    case tensorflow::DT_RESOURCE:
      return sizeof(tensorflow::ResourceHandle);
    default:
      return tensorflow::DataTypeSize(data_type);
  }
}

static void FuzzTest(
    const tensorflow::testing::CheckpointReaderFuzzInput& input) {
  // Using a ram file avoids disk I/O, speeding up the fuzzer.
  const std::string filename = "ram:///checkpoint";
  CreateCheckpoint(filename, input);
  // RamFileSystem::NewWritableFile doesn't remove existing files, so
  // expliciently ensure the checkpoint is deleted after each test.
  auto checkpoint_cleanup = tensorflow::gtl::MakeCleanup([&filename] {
    TF_CHECK_OK(tensorflow::Env::Default()->DeleteFile(filename));
  });

  tensorflow::TF_StatusPtr status(TF_NewStatus());
  tensorflow::checkpoint::CheckpointReader reader(filename, status.get());
  if (TF_GetCode(status.get()) != TF_OK) return;

  // Load each tensor in the input.
  std::unique_ptr<tensorflow::Tensor> tensor;
  for (const auto& entry : input.meta().tensor()) {
    // Fuzz tests have a memory limit of 2 GB; skipping tensors over 1 GB is
    // sufficient to avoid OOMs.
    static constexpr double kMaxTensorSize = 1e9;
    auto data_type = reader.GetVariableToDataTypeMap().find(entry.name());
    auto shape = reader.GetVariableToShapeMap().find(entry.name());
    if (data_type != reader.GetVariableToDataTypeMap().end() &&
        shape != reader.GetVariableToShapeMap().end() &&
        static_cast<double>(GetDataTypeSize(data_type->second)) *
                shape->second.num_elements() <
            kMaxTensorSize) {
      reader.GetTensor(entry.name(), &tensor, status.get());
    }
  }
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithSeeds(fuzztest::ReadFilesFromDirectory<
      tensorflow::testing::CheckpointReaderFuzzInput>(
        tensorflow::GetDataDependencyFilepath(
          "tensorflow/security/fuzzing/cc/checkpoint_reader_testdata")));

}  // namespace
