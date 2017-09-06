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

#include <vector>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/kernels/summary_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

Status SummaryTestHelper(
    const string& test_name,
    std::function<Status(SummaryWriterInterface*)> writer_fn,
    std::function<void(const Event&)> test_fn) {
  static std::set<string>* tests = new std::set<string>();
  CHECK(tests->insert(test_name).second) << ": " << test_name;

  SummaryWriterInterface* writer;
  Env* env = Env::Default();
  TF_CHECK_OK(
      CreateSummaryWriter(1, 1, testing::TmpDir(), test_name, env, &writer));
  core::ScopedUnref deleter(writer);

  TF_CHECK_OK(writer_fn(writer));
  TF_CHECK_OK(writer->Flush());

  std::vector<string> files;
  TF_CHECK_OK(env->GetChildren(testing::TmpDir(), &files));
  bool found = false;
  for (const string& f : files) {
    if (StringPiece(f).contains(test_name)) {
      if (found) {
        return errors::Unknown("Found more than one file for ", test_name);
      }
      found = true;
      std::unique_ptr<RandomAccessFile> read_file;
      TF_CHECK_OK(env->NewRandomAccessFile(io::JoinPath(testing::TmpDir(), f),
                                           &read_file));
      io::RecordReader reader(read_file.get(), io::RecordReaderOptions());
      string record;
      uint64 offset = 0;
      TF_CHECK_OK(reader.ReadRecord(&offset,
                                    &record));  // The first event is irrelevant
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      Event e;
      e.ParseFromString(record);
      test_fn(e);
    }
  }
  if (!found) {
    return errors::Unknown("Found no file for ", test_name);
  }
  return Status::OK();
}

TEST(SummaryInterfaceTest, WriteTensor) {
  TF_CHECK_OK(SummaryTestHelper("tensor_test",
                                [](SummaryWriterInterface* writer) {
                                  Tensor one(DT_FLOAT, TensorShape({}));
                                  one.scalar<float>()() = 1.0;
                                  TF_RETURN_IF_ERROR(writer->WriteTensor(
                                      2, one, "name",
                                      SummaryMetadata().SerializeAsString()));
                                  TF_RETURN_IF_ERROR(writer->Flush());
                                  return Status::OK();
                                },
                                [](const Event& e) {
                                  EXPECT_EQ(e.step(), 2);
                                  CHECK_EQ(e.summary().value_size(), 1);
                                  EXPECT_EQ(e.summary().value(0).tag(), "name");
                                }));
}

TEST(SummaryInterfaceTest, WriteScalar) {
  TF_CHECK_OK(SummaryTestHelper(
      "scalar_test",
      [](SummaryWriterInterface* writer) {
        Tensor one(DT_FLOAT, TensorShape({}));
        one.scalar<float>()() = 1.0;
        TF_RETURN_IF_ERROR(writer->WriteScalar(2, one, "name"));
        TF_RETURN_IF_ERROR(writer->Flush());
        return Status::OK();
      },
      [](const Event& e) {
        EXPECT_EQ(e.step(), 2);
        CHECK_EQ(e.summary().value_size(), 1);
        EXPECT_EQ(e.summary().value(0).tag(), "name");
        EXPECT_EQ(e.summary().value(0).simple_value(), 1.0);
      }));
}

TEST(SummaryInterfaceTest, WriteHistogram) {
  TF_CHECK_OK(SummaryTestHelper("hist_test",
                                [](SummaryWriterInterface* writer) {
                                  Tensor one(DT_FLOAT, TensorShape({}));
                                  one.scalar<float>()() = 1.0;
                                  TF_RETURN_IF_ERROR(
                                      writer->WriteHistogram(2, one, "name"));
                                  TF_RETURN_IF_ERROR(writer->Flush());
                                  return Status::OK();
                                },
                                [](const Event& e) {
                                  EXPECT_EQ(e.step(), 2);
                                  CHECK_EQ(e.summary().value_size(), 1);
                                  EXPECT_EQ(e.summary().value(0).tag(), "name");
                                  EXPECT_TRUE(e.summary().value(0).has_histo());
                                }));
}

TEST(SummaryInterfaceTest, WriteImage) {
  TF_CHECK_OK(SummaryTestHelper(
      "image_test",
      [](SummaryWriterInterface* writer) {
        Tensor one(DT_UINT8, TensorShape({1, 1, 1, 1}));
        one.scalar<int8>()() = 1;
        TF_RETURN_IF_ERROR(writer->WriteImage(2, one, "name", 1, Tensor()));
        TF_RETURN_IF_ERROR(writer->Flush());
        return Status::OK();
      },
      [](const Event& e) {
        EXPECT_EQ(e.step(), 2);
        CHECK_EQ(e.summary().value_size(), 1);
        EXPECT_EQ(e.summary().value(0).tag(), "name/image");
        CHECK(e.summary().value(0).has_image());
        EXPECT_EQ(e.summary().value(0).image().height(), 1);
        EXPECT_EQ(e.summary().value(0).image().width(), 1);
        EXPECT_EQ(e.summary().value(0).image().colorspace(), 1);
      }));
}

TEST(SummaryInterfaceTest, WriteAudio) {
  TF_CHECK_OK(SummaryTestHelper(
      "audio_test",
      [](SummaryWriterInterface* writer) {
        Tensor one(DT_FLOAT, TensorShape({1, 1}));
        one.scalar<float>()() = 1.0;
        TF_RETURN_IF_ERROR(writer->WriteAudio(2, one, "name", 1, 1));
        TF_RETURN_IF_ERROR(writer->Flush());
        return Status::OK();
      },
      [](const Event& e) {
        EXPECT_EQ(e.step(), 2);
        CHECK_EQ(e.summary().value_size(), 1);
        EXPECT_EQ(e.summary().value(0).tag(), "name/audio");
        CHECK(e.summary().value(0).has_audio());
      }));
}

}  // namespace
}  // namespace tensorflow
