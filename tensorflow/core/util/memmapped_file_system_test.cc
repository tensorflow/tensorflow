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
#include "tensorflow/core/util/memmapped_file_system.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/memmapped_file_system_writer.h"

namespace tensorflow {

namespace {

// Names of files in memmapped environment.
constexpr char kTensor1FileName[] = "memmapped_package://t1";
constexpr char kTensor2FileName[] = "memmapped_package://t2";
constexpr char kProtoFileName[] = "memmapped_package://b";
constexpr int kTestGraphDefVersion = 666;

Status CreateMemmappedFileSystemFile(const string& filename, bool corrupted,
                                     Tensor* test_tensor) {
  Env* env = Env::Default();
  MemmappedFileSystemWriter writer;
  TF_RETURN_IF_ERROR(writer.InitializeToFile(env, filename));

  // Try to write a tensor and proto.
  test::FillFn<float>(test_tensor,
                      [](int i) { return static_cast<float>(i * i); });

  TF_RETURN_IF_ERROR(writer.SaveTensor(*test_tensor, kTensor1FileName));

  // Create a proto with some fields.
  GraphDef graph_def;
  graph_def.set_version(kTestGraphDefVersion);
  TF_RETURN_IF_ERROR(writer.SaveProtobuf(graph_def, kProtoFileName));

  // Save a tensor after the proto to check that alignment works.
  test::FillFn<float>(test_tensor,
                      [](int i) { return static_cast<float>(i * i * i); });
  TF_RETURN_IF_ERROR(writer.SaveTensor(*test_tensor, kTensor2FileName));

  if (!corrupted) {
    // Flush and close the file.
    TF_RETURN_IF_ERROR(writer.FlushAndClose());
  }
  return Status::OK();
}

TEST(MemmappedFileSystemTest, SimpleTest) {
  const TensorShape test_tensor_shape = {10, 200};
  Tensor test_tensor(DT_FLOAT, test_tensor_shape);
  const string dir = testing::TmpDir();
  const string filename = io::JoinPath(dir, "memmapped_env_test");
  TF_ASSERT_OK(CreateMemmappedFileSystemFile(filename, false, &test_tensor));

  // Check that we can memmap the created file.
  MemmappedEnv memmapped_env(Env::Default());
  TF_ASSERT_OK(memmapped_env.InitializeFromFile(filename));
  // Try to load a proto from the file.
  GraphDef test_graph_def;
  TF_EXPECT_OK(
      ReadBinaryProto(&memmapped_env, kProtoFileName, &test_graph_def));
  EXPECT_EQ(kTestGraphDefVersion, test_graph_def.version());
  // Check that we can correctly get a tensor memory.
  std::unique_ptr<ReadOnlyMemoryRegion> memory_region;
  TF_ASSERT_OK(memmapped_env.NewReadOnlyMemoryRegionFromFile(kTensor2FileName,
                                                             &memory_region));

  // The memory region can be bigger but not less than Tensor size.
  ASSERT_GE(memory_region->length(), test_tensor.TotalBytes());
  EXPECT_EQ(test_tensor.tensor_data(),
            StringPiece(static_cast<const char*>(memory_region->data()),
                        test_tensor.TotalBytes()));
  // Check that GetFileSize works.
  uint64 file_size = 0;
  TF_ASSERT_OK(memmapped_env.GetFileSize(kTensor2FileName, &file_size));
  EXPECT_EQ(test_tensor.TotalBytes(), file_size);

  // Check that Stat works.
  FileStatistics stat;
  TF_ASSERT_OK(memmapped_env.Stat(kTensor2FileName, &stat));
  EXPECT_EQ(test_tensor.TotalBytes(), stat.length);

  // Check that if file not found correct error message returned.
  EXPECT_EQ(
      error::NOT_FOUND,
      memmapped_env.NewReadOnlyMemoryRegionFromFile("bla-bla", &memory_region)
          .code());

  // Check FileExists.
  TF_EXPECT_OK(memmapped_env.FileExists(kTensor2FileName));
  EXPECT_EQ(error::Code::NOT_FOUND,
            memmapped_env.FileExists("bla-bla-bla").code());
}

TEST(MemmappedFileSystemTest, NotInitalized) {
  MemmappedEnv memmapped_env(Env::Default());
  std::unique_ptr<ReadOnlyMemoryRegion> memory_region;
  EXPECT_EQ(
      error::FAILED_PRECONDITION,
      memmapped_env
          .NewReadOnlyMemoryRegionFromFile(kTensor1FileName, &memory_region)
          .code());
  std::unique_ptr<RandomAccessFile> file;
  EXPECT_EQ(error::FAILED_PRECONDITION,
            memmapped_env.NewRandomAccessFile(kProtoFileName, &file).code());
}

TEST(MemmappedFileSystemTest, Corrupted) {
  // Create a corrupted file (it is not closed it properly).
  const TensorShape test_tensor_shape = {100, 200};
  Tensor test_tensor(DT_FLOAT, test_tensor_shape);
  const string dir = testing::TmpDir();
  const string filename = io::JoinPath(dir, "memmapped_env_corrupted_test");
  TF_ASSERT_OK(CreateMemmappedFileSystemFile(filename, true, &test_tensor));
  MemmappedFileSystem memmapped_env;
  ASSERT_NE(memmapped_env.InitializeFromFile(Env::Default(), filename),
            Status::OK());
}

TEST(MemmappedFileSystemTest, ProxyToDefault) {
  MemmappedEnv memmapped_env(Env::Default());
  const string dir = testing::TmpDir();
  const string filename = io::JoinPath(dir, "test_file");
  // Check that we can create write and read ordinary file.
  std::unique_ptr<WritableFile> writable_file_temp;
  TF_ASSERT_OK(memmapped_env.NewAppendableFile(filename, &writable_file_temp));
  // Making sure to clean up after the test finishes.
  const auto adh = [&memmapped_env, &filename](WritableFile* f) {
      delete f;
      TF_CHECK_OK(memmapped_env.DeleteFile(filename));
  };
  std::unique_ptr<WritableFile, decltype(adh)> writable_file(
      writable_file_temp.release(), adh);
  const string test_string = "bla-bla-bla";
  TF_ASSERT_OK(writable_file->Append(test_string));
  TF_ASSERT_OK(writable_file->Close());
  uint64 file_length = 0;
  TF_EXPECT_OK(memmapped_env.GetFileSize(filename, &file_length));
  EXPECT_EQ(test_string.length(), file_length);
  FileStatistics stat;
  TF_EXPECT_OK(memmapped_env.Stat(filename, &stat));
  EXPECT_EQ(test_string.length(), stat.length);
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_ASSERT_OK(
      memmapped_env.NewRandomAccessFile(filename, &random_access_file));
}

}  // namespace
}  // namespace tensorflow
