/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tsl/lib/io/zstd/zstd_output_buffer.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/mock_writable_file.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace {
using ::testing::_;
using ::testing::AtLeast;
using ::testing::AtMost;
using ::testing::Gt;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::SizeIs;
using ::tsl::testing::IsOk;

TEST(ZstdOutputBufferTest, SimpleAppend) {
  tsl::MockWritableFile output_file{};
  std::string output;

  EXPECT_CALL(output_file, Append(_))
      .Times(AtLeast(1))
      .WillRepeatedly(Invoke([&output](absl::string_view data) {
        output.append(data);
        return absl::OkStatus();
      }));
  EXPECT_CALL(output_file, Close()).Times(1).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(output_file, Flush()).Times(AtLeast(0));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tsl::io::ZstdOutputBuffer> zstd_output_buffer,
      tsl::io::ZstdOutputBuffer::Create(&output_file, /*compression_level=*/3));
  EXPECT_THAT(zstd_output_buffer->Append("hello\n"), IsOk());
  EXPECT_THAT(zstd_output_buffer->Append("hello2\n"), IsOk());
  EXPECT_THAT(zstd_output_buffer->Close(), IsOk());
  EXPECT_THAT(output, SizeIs(Gt(0)));
}

TEST(ZstdOutputBufferTest, Flush) {
  tsl::MockWritableFile output_file{};

  EXPECT_CALL(output_file, Append(_)).WillRepeatedly(Return(absl::OkStatus()));

  // No flush or close during the Append.
  EXPECT_CALL(output_file, Flush()).Times(0);
  EXPECT_CALL(output_file, Close()).Times(0);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tsl::io::ZstdOutputBuffer> zstd_output_buffer,
      tsl::io::ZstdOutputBuffer::Create(&output_file, /*compression_level=*/3));
  EXPECT_THAT(zstd_output_buffer->Append("hello"), IsOk());
  ::testing::Mock::VerifyAndClear(&output_file);

  // Flush should be called once. Close should not be called.
  EXPECT_CALL(output_file, Append(_))
      .Times(AtLeast(1))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(output_file, Flush()).Times(1).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(output_file, Close()).Times(0);
  EXPECT_THAT(zstd_output_buffer->Flush(), IsOk());
  ::testing::Mock::VerifyAndClear(&output_file);

  // On close, flush may be called, and close should be called once.
  EXPECT_CALL(output_file, Flush())
      .Times(AtMost(1))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(output_file, Close()).Times(1).WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(zstd_output_buffer->Close(), IsOk());
  ::testing::Mock::VerifyAndClear(&output_file);
}

TEST(ZstdOutputBufferTest, AppendAfterFlush) {
  tsl::MockWritableFile output_file{};

  EXPECT_CALL(output_file, Append(_)).WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(output_file, Close()).Times(0);
  EXPECT_CALL(output_file, Flush()).Times(0);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tsl::io::ZstdOutputBuffer> zstd_output_buffer,
      tsl::io::ZstdOutputBuffer::Create(&output_file, /*compression_level=*/3));
  EXPECT_THAT(zstd_output_buffer->Append("hello"), IsOk());
  testing::Mock::VerifyAndClear(&output_file);

  EXPECT_CALL(output_file, Append(_))
      .Times(AtLeast(1))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(output_file, Flush()).Times(1);
  EXPECT_THAT(zstd_output_buffer->Flush(), IsOk());
  testing::Mock::VerifyAndClear(&output_file);

  EXPECT_CALL(output_file, Append(_))
      .Times(AtLeast(1))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_THAT(zstd_output_buffer->Append("hello"), IsOk());
  EXPECT_THAT(zstd_output_buffer->Close(), IsOk());
  testing::Mock::VerifyAndClear(&output_file);
}

TEST(ZstdOutputBufferTest, Sync) {
  tsl::MockWritableFile output_file{};

  // No flush or close during the Append.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tsl::io::ZstdOutputBuffer> zstd_output_buffer,
      tsl::io::ZstdOutputBuffer::Create(&output_file, /*compression_level=*/3));
  EXPECT_THAT(zstd_output_buffer->Append("hello"), IsOk());

  // When Sync is called, we expect one write call to the underlying file,
  // followed by a Sync call. Afterwards there may be another write call when
  // the stream gets closed.
  testing::Expectation append_action =
      EXPECT_CALL(output_file, Append(_))
          .Times(1)
          .WillRepeatedly(Return(absl::OkStatus()));
  testing::Expectation sync_action =
      EXPECT_CALL(output_file, Sync()).Times(1).After(append_action);
  EXPECT_CALL(output_file, Append(_))
      .Times(::testing::AtMost(1))
      .After(sync_action)
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_THAT(zstd_output_buffer->Sync(), IsOk());
}

}  // namespace
