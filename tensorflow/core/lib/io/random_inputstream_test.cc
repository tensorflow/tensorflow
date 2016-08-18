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

#include "tensorflow/core/lib/io/random_inputstream.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace io {
namespace {

TEST(RandomInputStream, ReadNBytes) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/random_inputbuffer_test";
  WriteStringToFile(env, fname, "0123456789");

  std::unique_ptr<RandomAccessFile> file;
  TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
  string read;
  RandomAccessInputStream in(file.get());
  TF_CHECK_OK(in.ReadNBytes(3, &read));
  EXPECT_EQ(read, "012");
  TF_CHECK_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "");
  TF_CHECK_OK(in.ReadNBytes(5, &read));
  EXPECT_EQ(read, "34567");
  TF_CHECK_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "");
  EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(20, &read)));
  EXPECT_EQ(read, "89");
  TF_CHECK_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "");
}

TEST(RandomInputStream, SkipNBytes) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/random_inputbuffer_test";
  WriteStringToFile(env, fname, "0123456789");

  std::unique_ptr<RandomAccessFile> file;
  TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
  string read;
  RandomAccessInputStream in(file.get());
  TF_CHECK_OK(in.SkipNBytes(3));
  TF_CHECK_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "");
  TF_CHECK_OK(in.ReadNBytes(4, &read));
  EXPECT_EQ(read, "3456");
  TF_CHECK_OK(in.SkipNBytes(0));
  TF_CHECK_OK(in.ReadNBytes(2, &read));
  EXPECT_EQ(read, "78");
  EXPECT_TRUE(errors::IsOutOfRange(in.SkipNBytes(20)));
  // Making sure that if we read after we've skipped beyond end of file, we get
  // nothing.
  EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
  EXPECT_EQ(read, "");
}

}  // anonymous namespace
}  // namespace io
}  // namespace tensorflow
