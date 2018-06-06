/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/packed_literal_reader.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class RoundTripPackedLiteralTest : public ClientLibraryTestBase {
 protected:
  // Sends the literal to the server and retrieves it back.
  std::unique_ptr<Literal> RoundTripToServer(const Literal& original) {
    std::unique_ptr<GlobalData> data =
        client_->TransferToServer(original).ConsumeValueOrDie();
    return client_->Transfer(*data).ConsumeValueOrDie();
  }
};

TEST_F(RoundTripPackedLiteralTest, RoundTripsR1F32Length2) {
  string data(sizeof(float) * 2, 0);
  tensorflow::gtl::MutableArraySlice<float> floats(
      tensorflow::bit_cast<float*>(data.data()), 2);
  floats[0] = 42.0;
  floats[1] = 24.0;

  string fname = tensorflow::testing::TmpDir() + "/RoundTripsR1F32Length2.data";
  EXPECT_TRUE(
      tensorflow::WriteStringToFile(tensorflow::Env::Default(), fname, data)
          .ok());

  std::unique_ptr<tensorflow::RandomAccessFile> f;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(fname, &f));
  PackedLiteralReader reader(f.release());
  std::unique_ptr<Literal> actual =
      reader.Read(ShapeUtil::MakeShape(F32, {2})).ConsumeValueOrDie();
  EXPECT_TRUE(reader.IsExhausted());

  EXPECT_EQ(42.0, actual->Get<float>({0}));
  EXPECT_EQ(24.0, actual->Get<float>({1}));
}

TEST_F(RoundTripPackedLiteralTest, RoundTripsR2F32Size2x2Dim0Minor) {
  string data(sizeof(float) * 4, 0);
  tensorflow::gtl::MutableArraySlice<float> floats(
      tensorflow::bit_cast<float*>(data.data()), 4);
  // With x as the minor dimension, these will become:
  floats[0] = 42.0;  // y=0,x=0
  floats[1] = 24.0;  // y=0,x=1
  floats[2] = 64.0;  // y=1,x=0
  floats[3] = 46.0;  // y=1,x=1

  string fname =
      tensorflow::testing::TmpDir() + "/RoundTripsR2F32Size2x2Dim0Minor.data";
  EXPECT_TRUE(
      tensorflow::WriteStringToFile(tensorflow::Env::Default(), fname, data)
          .ok());

  const Layout layout = LayoutUtil::MakeLayout({1, 0});

  std::unique_ptr<tensorflow::RandomAccessFile> f;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(fname, &f));
  PackedLiteralReader reader(f.release());
  std::unique_ptr<Literal> actual =
      reader.Read(ShapeUtil::MakeShape(F32, {2, 2}), &layout)
          .ConsumeValueOrDie();
  EXPECT_TRUE(reader.IsExhausted());

  EXPECT_EQ(42.0f, actual->Get<float>({0, 0}));
  EXPECT_EQ(24.0f, actual->Get<float>({0, 1}));
  EXPECT_EQ(64.0f, actual->Get<float>({1, 0}));
  EXPECT_EQ(46.0f, actual->Get<float>({1, 1}));

  std::unique_ptr<Literal> round_tripped = RoundTripToServer(*actual);
  EXPECT_TRUE(LiteralTestUtil::Equal(*round_tripped, *actual));
}

TEST_F(RoundTripPackedLiteralTest, RoundTripsR2F32Size2x2Dim1Minor) {
  string data(sizeof(float) * 4, 0);
  tensorflow::gtl::MutableArraySlice<float> floats(
      tensorflow::bit_cast<float*>(data.data()), 4);
  // With y as the minor dimension, these will become:
  floats[0] = 42.0;  // y=0,x=0
  floats[1] = 24.0;  // y=1,x=0
  floats[2] = 64.0;  // y=0,x=1
  floats[3] = 46.0;  // y=1,x=1

  string fname =
      tensorflow::testing::TmpDir() + "/RoundTripsR2F32Size2x2Dim1Minor.data";
  EXPECT_TRUE(
      tensorflow::WriteStringToFile(tensorflow::Env::Default(), fname, data)
          .ok());

  const Layout layout = LayoutUtil::MakeLayout({0, 1});

  std::unique_ptr<tensorflow::RandomAccessFile> f;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(fname, &f));
  PackedLiteralReader reader(f.release());
  std::unique_ptr<Literal> actual =
      reader.Read(ShapeUtil::MakeShape(F32, {2, 2}), &layout)
          .ConsumeValueOrDie();
  EXPECT_TRUE(reader.IsExhausted());

  EXPECT_EQ(42.0f, actual->Get<float>({0, 0}));
  EXPECT_EQ(24.0f, actual->Get<float>({1, 0}));
  EXPECT_EQ(64.0f, actual->Get<float>({0, 1}));
  EXPECT_EQ(46.0f, actual->Get<float>({1, 1}));

  std::unique_ptr<Literal> round_tripped = RoundTripToServer(*actual);
  EXPECT_TRUE(LiteralTestUtil::Equal(*round_tripped, *actual));
}

}  // namespace
}  // namespace xla
