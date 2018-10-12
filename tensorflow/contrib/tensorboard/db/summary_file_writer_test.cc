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
#include "tensorflow/contrib/tensorboard/db/summary_file_writer.h"

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

class FakeClockEnv : public EnvWrapper {
 public:
  FakeClockEnv() : EnvWrapper(Env::Default()), current_millis_(0) {}
  void AdvanceByMillis(const uint64 millis) { current_millis_ += millis; }
  uint64 NowMicros() override { return current_millis_ * 1000; }
  uint64 NowSeconds() override { return current_millis_ * 1000; }

 private:
  uint64 current_millis_;
};

class SummaryFileWriterTest : public ::testing::Test {
 protected:
  Status SummaryTestHelper(
      const string& test_name,
      const std::function<Status(SummaryWriterInterface*)>& writer_fn,
      const std::function<void(const Event&)>& test_fn) {
    static std::set<string>* tests = new std::set<string>();
    CHECK(tests->insert(test_name).second) << ": " << test_name;

    SummaryWriterInterface* writer;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 1, testing::TmpDir(), test_name,
                                        &env_, &writer));
    core::ScopedUnref deleter(writer);

    TF_CHECK_OK(writer_fn(writer));
    TF_CHECK_OK(writer->Flush());

    std::vector<string> files;
    TF_CHECK_OK(env_.GetChildren(testing::TmpDir(), &files));
    bool found = false;
    for (const string& f : files) {
      if (str_util::StrContains(f, test_name)) {
        if (found) {
          return errors::Unknown("Found more than one file for ", test_name);
        }
        found = true;
        std::unique_ptr<RandomAccessFile> read_file;
        TF_CHECK_OK(env_.NewRandomAccessFile(io::JoinPath(testing::TmpDir(), f),
                                             &read_file));
        io::RecordReader reader(read_file.get(), io::RecordReaderOptions());
        string record;
        uint64 offset = 0;
        TF_CHECK_OK(
            reader.ReadRecord(&offset,
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

  FakeClockEnv env_;
};

TEST_F(SummaryFileWriterTest, WriteTensor) {
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

TEST_F(SummaryFileWriterTest, WriteScalar) {
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

TEST_F(SummaryFileWriterTest, WriteHistogram) {
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

TEST_F(SummaryFileWriterTest, WriteImage) {
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

TEST_F(SummaryFileWriterTest, WriteAudio) {
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

TEST_F(SummaryFileWriterTest, WriteEvent) {
  TF_CHECK_OK(
      SummaryTestHelper("event_test",
                        [](SummaryWriterInterface* writer) {
                          std::unique_ptr<Event> e{new Event};
                          e->set_step(7);
                          e->mutable_summary()->add_value()->set_tag("hi");
                          TF_RETURN_IF_ERROR(writer->WriteEvent(std::move(e)));
                          TF_RETURN_IF_ERROR(writer->Flush());
                          return Status::OK();
                        },
                        [](const Event& e) {
                          EXPECT_EQ(e.step(), 7);
                          CHECK_EQ(e.summary().value_size(), 1);
                          EXPECT_EQ(e.summary().value(0).tag(), "hi");
                        }));
}

TEST_F(SummaryFileWriterTest, WallTime) {
  env_.AdvanceByMillis(7023);
  TF_CHECK_OK(SummaryTestHelper(
      "wall_time_test",
      [](SummaryWriterInterface* writer) {
        Tensor one(DT_FLOAT, TensorShape({}));
        one.scalar<float>()() = 1.0;
        TF_RETURN_IF_ERROR(writer->WriteScalar(2, one, "name"));
        TF_RETURN_IF_ERROR(writer->Flush());
        return Status::OK();
      },
      [](const Event& e) { EXPECT_EQ(e.wall_time(), 7.023); }));
}

}  // namespace
}  // namespace tensorflow
