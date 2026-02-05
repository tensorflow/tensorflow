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

#include "xla/util/split_proto/split_gpu_executable_writer.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util/split_proto/split_proto_reader.h"

namespace xla {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using ::xla::gpu::GpuExecutableProto;

TEST(SplitGpuExecutableWriterTest, WriteSplitGpuExecutable) {
  auto initialExecutable = ParseTextProtoOrDie<GpuExecutableProto>(R"pb(
    hlo_module_with_config { hlo_module { name: "test_module" } }
    buffer_allocations { values { index: 0 size: 2 } }
    asm_text: "asm_text"
    binary: "binary_data"
    dnn_compiled_graphs { key: "key" value: "value" }
    gpu_compute_capability {
      cuda_compute_capability { major: 9 minor: 0 feature_extension: NONE }
    }
    thunk { thunk_info { thunk_id: 1 } }
    module_name: "test_module"
    program_shape { parameter_names: [ "name1", "name2" ] }
    output_info_map {
      shape_index: { indexes: [ 1, 2 ] }
      output_info { allocation_index: 0 passthrough: true }
    }
    constants {
      symbol_name: "constant_1"
      content { data: "constant_data" }
      allocation_index: 0
    }
    constants {
      symbol_name: "constant_2"
      content { data: "constant_data" }
      allocation_index: 1
    }
  )pb");

  std::string serialized;
  auto writer = std::make_unique<riegeli::StringWriter<>>(&serialized);
  ASSERT_OK(WriteSplitGpuExecutable(initialExecutable, std::move(writer)));

  GpuExecutableProto deserializedExecutable;
  auto reader = std::make_unique<riegeli::StringReader<>>(serialized);
  ASSERT_OK(ReadSplitProto(std::move(reader), deserializedExecutable));

  EXPECT_THAT(deserializedExecutable, EqualsProto(initialExecutable));
}

}  // namespace
}  // namespace xla
