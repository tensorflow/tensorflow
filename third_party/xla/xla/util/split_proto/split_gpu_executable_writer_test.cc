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
#include "riegeli/base/maker.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using ::xla::gpu::GpuExecutableProto;

TEST(SplitGpuExecutableWriterTest, WriteSplitGpuExecutable) {
  auto initialExecutable = ParseTextProtoOrDie<GpuExecutableProto>(R"pb(
    hlo_module_with_config { hlo_module { id: 1 name: "test_module" } }
    buffer_allocations { values { index: 0 size: 2 } }
    asm_text: "asm_text"
    binary: "binary_data"
    dnn_compiled_graphs { key: "key" value: "value" }
    gpu_compute_capability {
      cuda_compute_capability { major: 9 minor: 0 feature_extension: NONE }
    }
    thunks { thunk_info { thunk_id: 1 } }
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

  // The module ID shouldn't be serialized.
  initialExecutable.mutable_hlo_module_with_config()
      ->mutable_hlo_module()
      ->clear_id();
  EXPECT_THAT(deserializedExecutable, EqualsProto(initialExecutable));
}

TEST(SplitGpuExecutableWriterTest, JsonBackendConfigIsNormalized) {
  GpuExecutableProto proto1;
  *proto1.mutable_hlo_module_with_config()
       ->mutable_hlo_module()
       ->add_computations()
       ->add_instructions()
       ->mutable_backend_config() = R"json({"a": 1, "b": 2, "c": 3})json";

  GpuExecutableProto proto2;
  *proto2.mutable_hlo_module_with_config()
       ->mutable_hlo_module()
       ->add_computations()
       ->add_instructions()
       ->mutable_backend_config() = R"json({"c": 3, "b": 2, "a": 1})json";

  std::string serialized1;
  ASSERT_OK(WriteSplitGpuExecutable(
      proto1, std::make_unique<riegeli::StringWriter<>>(&serialized1)));

  std::string serialized2;
  ASSERT_OK(WriteSplitGpuExecutable(
      proto2, std::make_unique<riegeli::StringWriter<>>(&serialized2)));

  EXPECT_EQ(serialized1, serialized2);
}

TEST(SplitGpuExecutableWriterTest, NonJsonBackendConfigIsAccepted) {
  GpuExecutableProto proto1;
  *proto1.mutable_hlo_module_with_config()
       ->mutable_hlo_module()
       ->add_computations()
       ->add_instructions()
       ->mutable_backend_config() = "x-json";

  std::string serialized1;
  ASSERT_OK(WriteSplitGpuExecutable(
      proto1, std::make_unique<riegeli::StringWriter<>>(&serialized1)));
}

TEST(SplitGpuExecutableWriterTest, JsonBackendConfigPayloadIsNormalized) {
  GpuExecutableProto proto1;
  proto1.mutable_hlo_module_with_config()
      ->mutable_hlo_module()
      ->add_computations()
      ->add_instructions()
      ->mutable_backend_config_payload()
      ->set_value(R"json({"a": 1, "b": 2, "c": 3})json");

  GpuExecutableProto proto2;
  proto2.mutable_hlo_module_with_config()
      ->mutable_hlo_module()
      ->add_computations()
      ->add_instructions()
      ->mutable_backend_config_payload()
      ->set_value(R"json({"c": 3, "b": 2, "a": 1})json");

  std::string serialized1;
  ASSERT_OK(WriteSplitGpuExecutable(
      proto1, std::make_unique<riegeli::StringWriter<>>(&serialized1)));

  std::string serialized2;
  ASSERT_OK(WriteSplitGpuExecutable(
      proto2, std::make_unique<riegeli::StringWriter<>>(&serialized2)));

  EXPECT_EQ(serialized1, serialized2);
}

TEST(SplitGpuExecutableWriterTest,
     JsonBackendConfigExternalPayloadIsNormalized) {
  GpuExecutableProto proto1;
  auto* module1 = proto1.mutable_hlo_module_with_config()->mutable_hlo_module();
  module1->add_payloads(R"json({"a": 1, "b": 2, "c": 3})json");
  module1->add_computations()
      ->add_instructions()
      ->mutable_backend_config_payload()
      ->set_id(0);

  GpuExecutableProto proto2;
  auto* module2 = proto2.mutable_hlo_module_with_config()->mutable_hlo_module();
  module2->add_payloads(R"json({"c": 3, "b": 2, "a": 1})json");
  module2->add_computations()
      ->add_instructions()
      ->mutable_backend_config_payload()
      ->set_id(0);

  std::string serialized1;
  ASSERT_OK(WriteSplitGpuExecutable(
      proto1, std::make_unique<riegeli::StringWriter<>>(&serialized1)));

  std::string serialized2;
  ASSERT_OK(WriteSplitGpuExecutable(
      proto2, std::make_unique<riegeli::StringWriter<>>(&serialized2)));

  EXPECT_EQ(serialized1, serialized2);
}

TEST(SplitGpuExecutableWriterTest,
     JsonBackendConfigDeduplicationIsDeterministic) {
  GpuExecutableProto proto;
  HloModuleProto* module =
      proto.mutable_hlo_module_with_config()->mutable_hlo_module();

  // Create 3 payloads, but 2 of them become identical after normalization.
  module->add_payloads(R"json({"a": 1, "b": 2})json");  // ID 0
  module->add_payloads(R"json({"x": 10})json");         // ID 1
  module->add_payloads(R"json({"b": 2, "a": 1})json");  // ID 2

  HloComputationProto* comp = module->add_computations();
  comp->add_instructions()->mutable_backend_config_payload()->set_id(0);
  comp->add_instructions()->mutable_backend_config_payload()->set_id(1);
  comp->add_instructions()->mutable_backend_config_payload()->set_id(2);

  std::string serialized;
  ASSERT_OK(WriteSplitGpuExecutable(
      proto, riegeli::Maker<riegeli::StringWriter>(&serialized)));

  GpuExecutableProto deserializedExecutable;
  ASSERT_OK(ReadSplitProto(riegeli::Maker<riegeli::StringReader>(serialized),
                           deserializedExecutable));

  HloModuleProto* read_module =
      deserializedExecutable.mutable_hlo_module_with_config()
          ->mutable_hlo_module();

  // Payloads should be deduplicated (only 2 unique payloads remain).
  EXPECT_EQ(read_module->payloads_size(), 2);

  // Instruction 0 and instruction 2 should now point to the same id.
  const HloComputationProto* read_comp = &read_module->computations(0);
  EXPECT_EQ(read_comp->instructions(0).backend_config_payload().id(),
            read_comp->instructions(2).backend_config_payload().id());

  // Instruction 1 should point to the other ID.
  EXPECT_NE(read_comp->instructions(0).backend_config_payload().id(),
            read_comp->instructions(1).backend_config_payload().id());
}

}  // namespace
}  // namespace xla
