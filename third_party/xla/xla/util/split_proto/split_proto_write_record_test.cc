/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/util/split_proto/split_proto_write_record.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "riegeli/bytes/string_writer.h"
#include "riegeli/records/record_writer.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/tsl/util/proto/parse_text_proto.h"

namespace xla {
namespace {

using tsl::proto_testing::ParseTextProtoOrDie;

TEST(SplitProtoWriteRecordTest, MapSerializationIsDeterministic) {
  auto proto1 = ParseTextProtoOrDie<ExecutableAndOptionsProto>(R"pb(
    compile_options {
      env_option_overrides {
        key: "a"
        value { string_field: "A" }
      }
      env_option_overrides {
        key: "b"
        value { string_field: "B" }
      }
      env_option_overrides {
        key: "c"
        value { string_field: "C" }
      }
    }
  )pb");

  auto proto2 = ParseTextProtoOrDie<ExecutableAndOptionsProto>(R"pb(
    compile_options {
      env_option_overrides {
        key: "c"
        value { string_field: "C" }
      }
      env_option_overrides {
        key: "b"
        value { string_field: "B" }
      }
      env_option_overrides {
        key: "a"
        value { string_field: "A" }
      }
    }
  )pb");

  std::string serialized1;
  riegeli::RecordWriter record_writer1(
      std::make_unique<riegeli::StringWriter<>>(&serialized1));
  ASSERT_OK(WriteRecord(record_writer1, proto1));

  std::string serialized2;
  riegeli::RecordWriter record_writer2(
      std::make_unique<riegeli::StringWriter<>>(&serialized2));
  ASSERT_OK(WriteRecord(record_writer2, proto2));

  EXPECT_EQ(serialized1, serialized2);
}

}  // namespace
}  // namespace xla
