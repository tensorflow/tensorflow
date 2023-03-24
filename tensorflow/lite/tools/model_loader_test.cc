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
#include "tensorflow/lite/tools/model_loader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace tools {
namespace {

static constexpr char kModelPath[] =
    "third_party/tensorflow/lite/java/demo/app/src/main/assets/"
    "mobilenet_v1_1.0_224.tflite";

using ::testing::IsNull;
using ::testing::Not;
using ::testing::WhenDynamicCastTo;

class ModelLoaderTest : public ::testing::Test {};

TEST_F(ModelLoaderTest, CreateFromModelPath) {
  auto model_loader = std::make_unique<PathModelLoader>(kModelPath);

  ASSERT_NE(model_loader, nullptr);
  EXPECT_TRUE(model_loader->Init());
}

TEST_F(ModelLoaderTest, CreateFromFdPath) {
  int fd = open(kModelPath, O_RDONLY);
  ASSERT_GE(fd, 0);
  struct stat stat_buf = {0};
  ASSERT_EQ(fstat(fd, &stat_buf), 0);
  auto model_loader =
      std::make_unique<MmapModelLoader>(fd, 0, stat_buf.st_size);
  close(fd);

  ASSERT_NE(model_loader, nullptr);
  EXPECT_TRUE(model_loader->Init());
}

TEST_F(ModelLoaderTest, CreateFromPipePath) {
  // Setup.
  // Read the model and serialize it.
  auto model = FlatBufferModel::BuildFromFile(kModelPath);
  flatbuffers::FlatBufferBuilder fbb;
  ModelT model_obj;
  model->GetModel()->UnPackTo(&model_obj);
  std::string model_description = model_obj.description;
  FinishModelBuffer(fbb, CreateModel(fbb, &model_obj));
  int pipe_fds[2];
  ASSERT_EQ(pipe(pipe_fds), 0);
  pid_t r = fork();
  // Child thread to write to pipe.
  if (r == 0) {
    close(pipe_fds[0]);
    int written_bytes = 0;
    int remaining_bytes = fbb.GetSize();
    uint8_t* buffer = fbb.GetBufferPointer();
    while (remaining_bytes > 0 &&
           (written_bytes = write(pipe_fds[1], buffer, remaining_bytes)) > 0) {
      remaining_bytes -= written_bytes;
      buffer += written_bytes;
    }
    close(pipe_fds[1]);
    ASSERT_TRUE(written_bytes > 0 && remaining_bytes == 0);
    _exit(0);
  }

  // Execute.
  // Parent thread.
  // Close the write pipe.
  close(pipe_fds[1]);
  auto model_loader =
      std::make_unique<PipeModelLoader>(pipe_fds[0], fbb.GetSize());
  ASSERT_NE(model_loader, nullptr);

  // Verify.
  EXPECT_TRUE(model_loader->Init());
  EXPECT_EQ(model_loader->GetModel()->GetModel()->description()->string_view(),
            model_description);
}

TEST_F(ModelLoaderTest, CreateBufferModelLoader) {
  auto model = FlatBufferModel::BuildFromFile(kModelPath);
  flatbuffers::FlatBufferBuilder fbb;
  ModelT model_obj;
  model->GetModel()->UnPackTo(&model_obj);
  std::string model_description = model_obj.description;
  FinishModelBuffer(fbb, CreateModel(fbb, &model_obj));
  ASSERT_NE(model->allocation(), nullptr);

  auto model_loader = std::make_unique<BufferModelLoader>(
      reinterpret_cast<const char*>(fbb.GetBufferPointer()), fbb.GetSize());
  ASSERT_NE(model_loader, nullptr);

  EXPECT_TRUE(model_loader->Init());
  EXPECT_EQ(model_loader->GetModel()->GetModel()->description()->string_view(),
            model_description);
}

TEST_F(ModelLoaderTest, InvalidModelPath) {
  auto model_loader = std::make_unique<PathModelLoader>("invalid/path");

  ASSERT_NE(model_loader, nullptr);
  EXPECT_FALSE(model_loader->Init());
}

TEST_F(ModelLoaderTest, InvalidFd) {
  auto model_loader = std::make_unique<MmapModelLoader>(0, 5, 10);

  ASSERT_NE(model_loader, nullptr);
  EXPECT_FALSE(model_loader->Init());
}

TEST_F(ModelLoaderTest, InvalidPipe) {
  auto model_loader = std::make_unique<PipeModelLoader>(-1, 10);

  ASSERT_NE(model_loader, nullptr);
  EXPECT_FALSE(model_loader->Init());
}

TEST_F(ModelLoaderTest, CreateModelLoaderFromValidPath) {
  EXPECT_THAT(CreateModelLoaderFromPath("a/b/c").get(),
              WhenDynamicCastTo<PathModelLoader*>(Not(IsNull())));
  EXPECT_THAT(CreateModelLoaderFromPath("fd:1:2:3").get(),
              WhenDynamicCastTo<MmapModelLoader*>(Not(IsNull())));
  EXPECT_THAT(CreateModelLoaderFromPath("pipe:1:2:3").get(),
              WhenDynamicCastTo<PipeModelLoader*>(Not(IsNull())));
  EXPECT_THAT(CreateModelLoaderFromPath("buffer:1:2").get(),
              WhenDynamicCastTo<BufferModelLoader*>(Not(IsNull())));
}

TEST_F(ModelLoaderTest, CreateModelLoaderFromInvalidPath) {
  EXPECT_EQ(CreateModelLoaderFromPath("fd:1"), nullptr);
  EXPECT_EQ(CreateModelLoaderFromPath("fd:1:2:3:4"), nullptr);

  EXPECT_EQ(CreateModelLoaderFromPath("pipe:1"), nullptr);
  EXPECT_EQ(CreateModelLoaderFromPath("pipe:1:2:3:4"), nullptr);
  EXPECT_EQ(CreateModelLoaderFromPath("buffer:1:2:3"), nullptr);
}

}  // namespace
}  // namespace tools
}  // namespace tflite
