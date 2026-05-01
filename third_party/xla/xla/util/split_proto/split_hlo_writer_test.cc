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

#include "xla/util/split_proto/split_hlo_writer.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/service/hlo.pb.h"
#include "xla/util/split_proto/split_proto_reader.h"

namespace xla {
namespace {

using ::tsl::proto_testing::EqualsProto;
using tsl::proto_testing::ParseTextProtoOrDie;

TEST(SplitHloWriterTest, WriteHloProto) {
  auto initial_proto = ParseTextProtoOrDie<HloProto>(R"pb(
    hlo_module {
      name: "test_module"
      computations {
        name: "test_computation_1"
        instructions {
          name: "test_instruction"
          opcode: "add"
          operand_ids: [0]
        }
      }
      computations {
        name: "test_computation_2"
        instructions {
          name: "test_instruction"
          opcode: "add"
          operand_ids: [0]
        }
      }
    }
    buffer_assignment {
      logical_buffers {
        id: 1
        size: 1024
        color: 0
      }
    }
  )pb");

  std::string serialized;
  auto writer = std::make_unique<riegeli::StringWriter<>>(&serialized);
  ASSERT_OK(WriteSplitHloProto(initial_proto, std::move(writer)));

  HloProto deserialized_proto;
  auto reader = std::make_unique<riegeli::StringReader<>>(serialized);
  ASSERT_OK(ReadSplitProto(std::move(reader), deserialized_proto));

  EXPECT_THAT(deserialized_proto, EqualsProto(initial_proto));
}

}  // namespace
}  // namespace xla
