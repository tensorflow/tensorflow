/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/utils.h"

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/core/data/file_logger_client_interface.h"
#include "tensorflow/core/data/file_logger_client_no_op.h"

namespace tensorflow::data {
namespace {

TEST(Util, CreateFileLoggerClient) {
  std::unique_ptr<FileLoggerClientInterface> client = CreateFileLoggerClient();
  EXPECT_NE(dynamic_cast<FileLoggerClientNoOp*>(client.get()), nullptr);
}

TEST(Util, DefaultDataTransferProtocol) {
  EXPECT_EQ(DefaultDataTransferProtocol(), "grpc");
}

TEST(TranslateFileName, NoOp) {
  constexpr char file[] = "/home/tfdata/file1";
  EXPECT_EQ(TranslateFileName(file), file);
}

TEST(TranslateFileName, EmptyPath) {
  constexpr char file[] = "";
  EXPECT_EQ(TranslateFileName(file), file);
}

TEST(TranslateFileName, TfDataPath) {
  constexpr char file[] = "tfdata/file1";
  EXPECT_EQ(TranslateFileName(file), file);
}

TEST(LocalityOptimizedPath, NoOp) {
  constexpr char file[] = "/home/tfdata/file1";
  EXPECT_EQ(LocalityOptimizedPath(file), file);
}

TEST(LocalityOptimizedPath, EmptyPath) {
  constexpr char file[] = "";
  EXPECT_EQ(LocalityOptimizedPath(file), file);
}

TEST(LocalityOptimizedPath, TfDataPath) {
  constexpr char file[] = "tfdata/file1";
  EXPECT_EQ(LocalityOptimizedPath(file), file);
}

}  // namespace
}  // namespace tensorflow::data
