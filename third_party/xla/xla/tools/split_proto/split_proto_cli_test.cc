/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "google/protobuf/text_format.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/bytes/writer.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/gpu/dense_data_intermediate.pb.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/tools/split_proto/split_proto_cli_lib.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::split_proto_cli {
namespace {

using ::absl_testing::StatusIs;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using ::tsl::proto_testing::Partially;

TEST(SplitProtoCliTest, PackAndUnpackGpuExecutableProtoRoundTrip) {
  auto initial_proto = ParseTextProtoOrDie<gpu::GpuExecutableProto>(R"pb(
    asm_text: "some_asm_text"
    binary: "some_binary"
    constants {
      symbol_name: "const_1"
      content { data: "const_1_data" }
      allocation_index: 0
    }
  )pb");

  std::string text_input;
  ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(initial_proto, &text_input));

  // 1. Pack text input to split proto
  std::unique_ptr<riegeli::Reader> text_reader =
      riegeli::Maker<riegeli::StringReader>(text_input);
  std::string split_bytes;
  std::unique_ptr<riegeli::Writer> split_writer =
      riegeli::Maker<riegeli::StringWriter>(&split_bytes);
  PackOptions pack_opts;
  pack_opts.proto_type = "xla.gpu.GpuExecutableProto";
  pack_opts.input_format = ProtoFormat::kText;

  ASSERT_OK(Pack(std::move(text_reader), std::move(split_writer), pack_opts));

  // 2. Unpack split proto to text output
  std::unique_ptr<riegeli::Reader> split_reader =
      riegeli::Maker<riegeli::StringReader>(split_bytes);
  std::string text_output;
  std::unique_ptr<riegeli::Writer> text_output_writer =
      riegeli::Maker<riegeli::StringWriter>(&text_output);
  UnpackOptions unpack_opts;
  unpack_opts.output_format = ProtoFormat::kText;

  ASSERT_OK(Unpack(std::move(split_reader), std::move(text_output_writer),
                   unpack_opts));

  // 3. Verify
  gpu::GpuExecutableProto final_proto =
      ParseTextProtoOrDie<gpu::GpuExecutableProto>(text_output);
  EXPECT_THAT(final_proto, Partially(EqualsProto(initial_proto)));
}

TEST(SplitProtoCliTest, PackAndUnpackExecutableAndOptionsProtoRoundTrip) {
  auto initial_proto = ParseTextProtoOrDie<ExecutableAndOptionsProto>(R"pb(
    serialized_executable: "some_serialized_executable"
    compile_options {
      executable_build_options {
        device_ordinal: 2
        result_layout { dimensions: [ 1, 2, 3 ] }
      }
    }
  )pb");

  std::string text_input;
  ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(initial_proto, &text_input));

  // 1. Pack text input to split proto
  std::unique_ptr<riegeli::Reader> text_reader =
      riegeli::Maker<riegeli::StringReader>(text_input);
  std::string split_bytes;
  std::unique_ptr<riegeli::Writer> split_writer =
      riegeli::Maker<riegeli::StringWriter>(&split_bytes);
  PackOptions pack_opts;
  pack_opts.proto_type = "xla.ExecutableAndOptionsProto";
  pack_opts.input_format = ProtoFormat::kText;

  ASSERT_OK(Pack(std::move(text_reader), std::move(split_writer), pack_opts));

  // 2. Unpack split proto to text output
  std::unique_ptr<riegeli::Reader> split_reader =
      riegeli::Maker<riegeli::StringReader>(split_bytes);
  std::string text_output;
  std::unique_ptr<riegeli::Writer> text_output_writer =
      riegeli::Maker<riegeli::StringWriter>(&text_output);
  UnpackOptions unpack_opts;
  unpack_opts.output_format = ProtoFormat::kText;

  ASSERT_OK(Unpack(std::move(split_reader), std::move(text_output_writer),
                   unpack_opts));

  // 3. Verify
  ExecutableAndOptionsProto final_proto =
      ParseTextProtoOrDie<ExecutableAndOptionsProto>(text_output);
  EXPECT_THAT(final_proto, Partially(EqualsProto(initial_proto)));
}

TEST(SplitProtoCliTest, PackInvalidProtoType) {
  std::unique_ptr<riegeli::Reader> reader =
      riegeli::Maker<riegeli::StringReader>("foo: 1");
  std::string split_bytes;
  std::unique_ptr<riegeli::Writer> writer =
      riegeli::Maker<riegeli::StringWriter>(&split_bytes);
  PackOptions pack_opts;
  pack_opts.proto_type = "xla.InvalidTypeProto";

  EXPECT_THAT(Pack(std::move(reader), std::move(writer), pack_opts),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::split_proto_cli
