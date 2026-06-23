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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/gpu/dense_data_intermediate.pb.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/tools/split_proto/split_proto_cli.pb.h"
#include "xla/tools/split_proto/split_proto_cli_lib.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

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
  std::string split_bytes;
  PackOptions pack_opts;
  pack_opts.proto_type = "xla.gpu.GpuExecutableProto";
  pack_opts.input_format = ProtoFormat::kText;

  ASSERT_OK(Pack(riegeli::Maker<riegeli::StringReader>(text_input),
                 riegeli::Maker<riegeli::StringWriter>(&split_bytes),
                 pack_opts));

  // 2. Unpack split proto to text output
  std::string text_output;
  UnpackOptions unpack_opts;
  unpack_opts.output_format = ProtoFormat::kText;

  ASSERT_OK(Unpack(riegeli::Maker<riegeli::StringReader>(split_bytes),
                   riegeli::Maker<riegeli::StringWriter>(&text_output),
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
  std::string split_bytes;
  PackOptions pack_opts;
  pack_opts.proto_type = "xla.ExecutableAndOptionsProto";
  pack_opts.input_format = ProtoFormat::kText;

  ASSERT_OK(Pack(riegeli::Maker<riegeli::StringReader>(text_input),
                 riegeli::Maker<riegeli::StringWriter>(&split_bytes),
                 pack_opts));

  // 2. Unpack split proto to text output
  std::string text_output;
  UnpackOptions unpack_opts;
  unpack_opts.output_format = ProtoFormat::kText;

  ASSERT_OK(Unpack(riegeli::Maker<riegeli::StringReader>(split_bytes),
                   riegeli::Maker<riegeli::StringWriter>(&text_output),
                   unpack_opts));

  // 3. Verify
  ExecutableAndOptionsProto final_proto =
      ParseTextProtoOrDie<ExecutableAndOptionsProto>(text_output);
  EXPECT_THAT(final_proto, Partially(EqualsProto(initial_proto)));
}

TEST(SplitProtoCliTest, PackInvalidProtoType) {
  std::string split_bytes;
  PackOptions pack_opts;
  pack_opts.proto_type = "xla.InvalidTypeProto";

  EXPECT_THAT(
      Pack(riegeli::Maker<riegeli::StringReader>("foo: 1"),
           riegeli::Maker<riegeli::StringWriter>(&split_bytes), pack_opts),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SplitProtoCliTest, PackAotSpecialized) {
  auto initial_proto =
      ParseTextProtoOrDie<DeserializedSplitExecutableAndOptions>(R"pb(
        executable_and_options {
          compile_options { executable_build_options { device_ordinal: 2 } }
        }
        gpu_executable { asm_text: "some_asm_text" binary: "some_binary" }
      )pb");

  std::string text_input;
  ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(initial_proto, &text_input));

  std::string split_bytes;
  PackOptions pack_opts;
  pack_opts.input_format = ProtoFormat::kText;
  ASSERT_OK(PackAot(riegeli::Maker<riegeli::StringReader>(text_input),
                    riegeli::Maker<riegeli::StringWriter>(&split_bytes),
                    pack_opts));

  ExecutableAndOptionsProto final_outer_proto;
  ASSERT_OK(ReadSplitProto(riegeli::Maker<riegeli::StringReader>(split_bytes),
                           final_outer_proto));

  EXPECT_THAT(final_outer_proto.compile_options(),
              Partially(EqualsProto(
                  initial_proto.executable_and_options().compile_options())));

  gpu::GpuExecutableProto final_inner_proto;
  ASSERT_OK(ReadSplitProto(riegeli::Maker<riegeli::StringReader>(
                               final_outer_proto.serialized_executable()),
                           final_inner_proto));

  EXPECT_THAT(final_inner_proto,
              Partially(EqualsProto(initial_proto.gpu_executable())));
}

TEST(SplitProtoCliTest, UnpackAotSpecialized) {
  auto initial_outer_proto =
      ParseTextProtoOrDie<ExecutableAndOptionsProto>(R"pb(
        compile_options { executable_build_options { device_ordinal: 2 } }
      )pb");
  auto initial_inner_proto = ParseTextProtoOrDie<gpu::GpuExecutableProto>(R"pb(
    asm_text: "some_asm_text"
    binary: "some_binary"
  )pb");

  ASSERT_OK(WriteSplitGpuExecutable(
      initial_inner_proto,
      riegeli::Maker<riegeli::StringWriter>(
          initial_outer_proto.mutable_serialized_executable())));

  std::string packed_split_outer_proto;
  ASSERT_OK(WriteSplitExecutableAndOptions(
      initial_outer_proto,
      riegeli::Maker<riegeli::StringWriter>(&packed_split_outer_proto)));

  std::string text_output;
  UnpackOptions unpack_opts;
  unpack_opts.output_format = ProtoFormat::kText;
  ASSERT_OK(UnpackAot(
      riegeli::Maker<riegeli::StringReader>(packed_split_outer_proto),
      riegeli::Maker<riegeli::StringWriter>(&text_output), unpack_opts));

  DeserializedSplitExecutableAndOptions final_proto =
      ParseTextProtoOrDie<DeserializedSplitExecutableAndOptions>(text_output);
  EXPECT_THAT(final_proto.gpu_executable(),
              Partially(EqualsProto(initial_inner_proto)));
  EXPECT_THAT(final_proto.executable_and_options().compile_options(),
              Partially(EqualsProto(initial_outer_proto.compile_options())));
}

TEST(SplitProtoCliTest, PackAndUnpackAotBinaryRoundTrip) {
  auto initial_outer_proto =
      ParseTextProtoOrDie<ExecutableAndOptionsProto>(R"pb(
        compile_options { executable_build_options { device_ordinal: 2 } }
      )pb");
  auto initial_inner_proto = ParseTextProtoOrDie<gpu::GpuExecutableProto>(R"pb(
    asm_text: "some_asm_text"
    binary: "some_binary"
  )pb");

  ASSERT_OK(WriteSplitGpuExecutable(
      initial_inner_proto,
      riegeli::Maker<riegeli::StringWriter>(
          initial_outer_proto.mutable_serialized_executable())));

  std::string packed_split_outer_proto;
  ASSERT_OK(WriteSplitExecutableAndOptions(
      initial_outer_proto,
      riegeli::Maker<riegeli::StringWriter>(&packed_split_outer_proto)));

  std::string binary_output;
  UnpackOptions unpack_opts;
  unpack_opts.output_format = ProtoFormat::kBinary;
  ASSERT_OK(UnpackAot(
      riegeli::Maker<riegeli::StringReader>(packed_split_outer_proto),
      riegeli::Maker<riegeli::StringWriter>(&binary_output), unpack_opts));

  std::string repacked_split_outer_proto;
  PackOptions pack_opts;
  pack_opts.input_format = ProtoFormat::kBinary;
  ASSERT_OK(PackAot(
      riegeli::Maker<riegeli::StringReader>(binary_output),
      riegeli::Maker<riegeli::StringWriter>(&repacked_split_outer_proto),
      pack_opts));

  std::string text_output;
  unpack_opts.output_format = ProtoFormat::kText;
  ASSERT_OK(UnpackAot(
      riegeli::Maker<riegeli::StringReader>(repacked_split_outer_proto),
      riegeli::Maker<riegeli::StringWriter>(&text_output), unpack_opts));

  DeserializedSplitExecutableAndOptions final_proto =
      ParseTextProtoOrDie<DeserializedSplitExecutableAndOptions>(text_output);
  EXPECT_THAT(final_proto.gpu_executable(),
              Partially(EqualsProto(initial_inner_proto)));
  EXPECT_THAT(final_proto.executable_and_options().compile_options(),
              Partially(EqualsProto(initial_outer_proto.compile_options())));
}

TEST(SplitProtoCliTest, DiffEqualProtos) {
  auto initial_proto = ParseTextProtoOrDie<ExecutableAndOptionsProto>(R"pb(
    serialized_executable: "some_serialized_executable"
  )pb");

  std::string split_bytes1;
  PackOptions pack_opts;
  pack_opts.proto_type = "xla.ExecutableAndOptionsProto";
  pack_opts.input_format = ProtoFormat::kText;
  std::string text_input;
  ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(initial_proto, &text_input));
  ASSERT_OK(Pack(riegeli::Maker<riegeli::StringReader>(text_input),
                 riegeli::Maker<riegeli::StringWriter>(&split_bytes1),
                 pack_opts));

  std::string split_bytes2 = split_bytes1;

  std::string diff_output;
  ASSERT_OK(Diff(riegeli::Maker<riegeli::StringReader>(split_bytes1),
                 riegeli::Maker<riegeli::StringReader>(split_bytes2),
                 riegeli::Maker<riegeli::StringWriter>(&diff_output)));
  EXPECT_TRUE(diff_output.empty());
}

TEST(SplitProtoCliTest, DiffDifferentProtos) {
  auto proto1 = ParseTextProtoOrDie<ExecutableAndOptionsProto>(R"pb(
    serialized_executable: "some_serialized_executable"
  )pb");
  auto proto2 = ParseTextProtoOrDie<ExecutableAndOptionsProto>(R"pb(
    serialized_executable: "different_serialized_executable"
  )pb");

  std::string split_bytes1, split_bytes2;
  PackOptions pack_opts;
  pack_opts.proto_type = "xla.ExecutableAndOptionsProto";
  pack_opts.input_format = ProtoFormat::kText;
  std::string text_input1, text_input2;
  ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(proto1, &text_input1));
  ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(proto2, &text_input2));
  ASSERT_OK(Pack(riegeli::Maker<riegeli::StringReader>(text_input1),
                 riegeli::Maker<riegeli::StringWriter>(&split_bytes1),
                 pack_opts));
  ASSERT_OK(Pack(riegeli::Maker<riegeli::StringReader>(text_input2),
                 riegeli::Maker<riegeli::StringWriter>(&split_bytes2),
                 pack_opts));

  std::string diff_output;
  ASSERT_OK(Diff(riegeli::Maker<riegeli::StringReader>(split_bytes1),
                 riegeli::Maker<riegeli::StringReader>(split_bytes2),
                 riegeli::Maker<riegeli::StringWriter>(&diff_output)));
  EXPECT_FALSE(diff_output.empty());
  EXPECT_THAT(diff_output,
              ::testing::HasSubstr("different_serialized_executable"));
}

}  // namespace
}  // namespace xla::split_proto_cli
