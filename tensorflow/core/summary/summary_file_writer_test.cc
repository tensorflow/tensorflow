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
#include "tensorflow/core/summary/summary_file_writer.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

class FakeClockEnv : public EnvWrapper {
 public:
  FakeClockEnv() : EnvWrapper(Env::Default()), current_millis_(0) {}
  void AdvanceByMillis(const uint64 millis) { current_millis_ += millis; }
  uint64 NowMicros() const override { return current_millis_ * 1000; }
  uint64 NowSeconds() const override { return current_millis_ * 1000; }

 private:
  uint64 current_millis_;
};

class SummaryFileWriterTest : public ::testing::Test {
 protected:
  absl::Status SummaryTestHelper(
      const string& test_name,
      const std::function<absl::Status(SummaryWriterInterface*)>& writer_fn,
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
      if (absl::StrContains(f, test_name)) {
        if (found) {
          return errors::Unknown("Found more than one file for ", test_name);
        }
        found = true;
        std::unique_ptr<RandomAccessFile> read_file;
        TF_CHECK_OK(env_.NewRandomAccessFile(io::JoinPath(testing::TmpDir(), f),
                                             &read_file));
        io::RecordReader reader(read_file.get(), io::RecordReaderOptions());
        tstring record;
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
    return absl::OkStatus();
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
                                  return absl::OkStatus();
                                },
                                [](const Event& e) {
                                  EXPECT_EQ(e.step(), 2);
                                  CHECK_EQ(e.summary().value_size(), 1);
                                  EXPECT_EQ(e.summary().value(0).tag(), "name");
                                }));
  TF_CHECK_OK(SummaryTestHelper(
      "string_tensor_test",
      [](SummaryWriterInterface* writer) {
        Tensor hello(DT_STRING, TensorShape({}));
        hello.scalar<tstring>()() = "hello";
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            2, hello, "name", SummaryMetadata().SerializeAsString()));
        TF_RETURN_IF_ERROR(writer->Flush());
        return absl::OkStatus();
      },
      [](const Event& e) {
        EXPECT_EQ(e.step(), 2);
        CHECK_EQ(e.summary().value_size(), 1);
        EXPECT_EQ(e.summary().value(0).tag(), "name");
        EXPECT_EQ(e.summary().value(0).tensor().dtype(), DT_STRING);
        EXPECT_EQ(e.summary().value(0).tensor().string_val()[0], "hello");
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
        return absl::OkStatus();
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
                                  return absl::OkStatus();
                                },
                                [](const Event& e) {
                                  EXPECT_EQ(e.step(), 2);
                                  CHECK_EQ(e.summary().value_size(), 1);
                                  EXPECT_EQ(e.summary().value(0).tag(), "name");
                                  EXPECT_TRUE(e.summary().value(0).has_histo());
                                }));
}

namespace {

// Create a 1x1 monochrome image consisting of a single pixel oof the given
// type.
template <typename T>
static absl::Status CreateImage(SummaryWriterInterface* writer) {
  Tensor bad_color(DT_UINT8, TensorShape({1}));
  bad_color.scalar<uint8>()() = 0;
  Tensor one(DataTypeToEnum<T>::v(), TensorShape({1, 1, 1, 1}));
  one.scalar<T>()() = T(1);
  TF_RETURN_IF_ERROR(writer->WriteImage(2, one, "name", 1, bad_color));
  TF_RETURN_IF_ERROR(writer->Flush());
  return absl::OkStatus();
}

// Verify that the event contains an image generated by CreateImage above.
static void CheckImage(const Event& e) {
  EXPECT_EQ(e.step(), 2);
  CHECK_EQ(e.summary().value_size(), 1);
  EXPECT_EQ(e.summary().value(0).tag(), "name/image");
  CHECK(e.summary().value(0).has_image());
  EXPECT_EQ(e.summary().value(0).image().height(), 1);
  EXPECT_EQ(e.summary().value(0).image().width(), 1);
  EXPECT_EQ(e.summary().value(0).image().colorspace(), 1);
}

}  // namespace

TEST_F(SummaryFileWriterTest, WriteImageUInt8) {
  TF_CHECK_OK(
      SummaryTestHelper("image_test_uint8", CreateImage<uint8>, CheckImage));
}

TEST_F(SummaryFileWriterTest, WriteImageFloat) {
  TF_CHECK_OK(
      SummaryTestHelper("image_test_float", CreateImage<float>, CheckImage));
}

TEST_F(SummaryFileWriterTest, WriteImageHalf) {
  TF_CHECK_OK(SummaryTestHelper("image_test_half", CreateImage<Eigen::half>,
                                CheckImage));
}

TEST_F(SummaryFileWriterTest, WriteImageDouble) {
  TF_CHECK_OK(
      SummaryTestHelper("image_test_double", CreateImage<double>, CheckImage));
}

TEST_F(SummaryFileWriterTest, WriteAudio) {
  TF_CHECK_OK(SummaryTestHelper(
      "audio_test",
      [](SummaryWriterInterface* writer) {
        Tensor one(DT_FLOAT, TensorShape({1, 1}));
        one.scalar<float>()() = 1.0;
        TF_RETURN_IF_ERROR(writer->WriteAudio(2, one, "name", 1, 1));
        TF_RETURN_IF_ERROR(writer->Flush());
        return absl::OkStatus();
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
                          return absl::OkStatus();
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
        return absl::OkStatus();
      },
      [](const Event& e) { EXPECT_EQ(e.wall_time(), 7.023); }));
}

TEST_F(SummaryFileWriterTest, AvoidFilenameCollision) {
  // Keep unique with all other test names in this file.
  string test_name = "avoid_filename_collision_test";
  int num_files = 10;
  for (int i = 0; i < num_files; i++) {
    SummaryWriterInterface* writer;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 1, testing::TmpDir(), test_name,
                                        &env_, &writer));
    core::ScopedUnref deleter(writer);
  }
  std::vector<string> files;
  TF_CHECK_OK(env_.GetChildren(testing::TmpDir(), &files));
  // Filter `files` down to just those generated in this test.
  files.erase(std::remove_if(files.begin(), files.end(),
                             [test_name](string f) {
                               return !absl::StrContains(f, test_name);
                             }),
              files.end());
  EXPECT_EQ(num_files, files.size())
      << "files = [" << absl::StrJoin(files, ", ") << "]";
}

}  // namespace
}  // namespace tensorflow
