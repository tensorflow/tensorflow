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

#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "google/protobuf/text_format.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/dense_data_intermediate.pb.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tools/split_proto/split_proto_cli.pb.h"
#include "xla/tools/split_proto/split_proto_cli_lib.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla_data.pb.h"

namespace xla::split_proto_cli {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;
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

TEST(SplitProtoCliTest, PackAndUnpackHloProtoRoundTrip) {
  auto initial_proto = ParseTextProtoOrDie<HloProto>(R"pb(
    hlo_module {
      name: "some_module"
      entry_computation_name: "entry"
      computations {
        name: "entry"
        instructions {
          name: "parameter.0"
          opcode: "parameter"
          shape {
            element_type: F32
            dimensions: [ 2, 3 ]
          }
        }
      }
    }
  )pb");

  std::string text_input;
  ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(initial_proto, &text_input));

  // 1. Pack text input to split proto
  std::string split_bytes;
  PackOptions pack_opts;
  pack_opts.proto_type = "xla.HloProto";
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
  HloProto final_proto = ParseTextProtoOrDie<HloProto>(text_output);
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

TEST(SplitProtoCliTest, InfoCommandPrintsCorrectDetails) {
  xla::CustomCallTargetRegistry::Global()->Register(
      "my_custom_call_456", reinterpret_cast<void*>(0x1234), "CUDA");

  auto gpu_exec = ParseTextProtoOrDie<gpu::GpuExecutableProto>(R"pb(
    module_name: "test_module"
    gpu_compute_capability { cuda_compute_capability { major: 8 minor: 0 } }
    buffer_allocations { values { index: 0 size: 1024 } }
    thunks {
      kernel_thunk {
        kernel_name: "my_kernel_123"
        launch_dimensions {
          block_counts { coordinates { x: 1 y: 1 z: 1 } }
          thread_counts_per_block { coordinates { x: 1 y: 1 z: 1 } }
        }
      }
    }
    thunks {
      copy_thunk {
        source_buffer {
          slice { offset: 0 size: 4 buffer_allocation_index: 0 }
          shape { element_type: F32 }
        }
        destination_buffer {
          slice { offset: 4 size: 4 buffer_allocation_index: 0 }
          shape { element_type: F32 }
        }
      }
    }
    thunks {
      async_start_thunk {
        async_execution_id: 1
        computation_stream_id: 0
        thunks {
          thunks {
            kernel_thunk {
              kernel_name: "async_nested_kernel_111"
              launch_dimensions {
                block_counts { coordinates { x: 1 y: 1 z: 1 } }
                thread_counts_per_block { coordinates { x: 1 y: 1 z: 1 } }
              }
            }
          }
        }
      }
    }
    thunks { async_done_thunk { async_execution_id: 1 } }
    thunks {
      custom_call_thunk {
        target_name: "my_custom_call_456"
        api_version: API_VERSION_STATUS_RETURNING
      }
    }
    thunks {
      custom_kernel_thunk {
        custom_kernel {
          name: "my_custom_kernel_789"
          kernel_spec {
            ptx { data: "dummy_ptx_code" }
            kernel_name: "my_custom_kernel_789"
          }
          block_dims { coordinates { x: 1 y: 1 z: 1 } }
          thread_dims { coordinates { x: 1 y: 1 z: 1 } }
        }
      }
    }
    thunks {
      sequential_thunk {
        thunks {
          kernel_thunk {
            kernel_name: "nested_kernel.999"
            launch_dimensions {
              block_counts { coordinates { x: 1 y: 1 z: 1 } }
              thread_counts_per_block { coordinates { x: 1 y: 1 z: 1 } }
            }
          }
        }
      }
    }
  )pb");

  ExecutableAndOptionsProto initial_proto;
  ASSERT_OK(WriteSplitGpuExecutable(
      gpu_exec, riegeli::Maker<riegeli::StringWriter>(
                    initial_proto.mutable_serialized_executable())));
  *initial_proto.mutable_compile_options() =
      ParseTextProtoOrDie<CompileOptionsProto>(R"pb(
        target_config { gpu_device_info { threads_per_block_limit: 1024 } }
      )pb");

  std::string serialized_exec_and_opts;
  ASSERT_OK(WriteSplitExecutableAndOptions(
      initial_proto,
      riegeli::Maker<riegeli::StringWriter>(&serialized_exec_and_opts)));

  std::stringstream buffer;
  std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
  absl::Cleanup restore_cout = [old] { std::cout.rdbuf(old); };

  ASSERT_OK(
      AotInfo(riegeli::Maker<riegeli::StringReader>(serialized_exec_and_opts)));

  std::string output = buffer.str();
  EXPECT_THAT(output, HasSubstr("Module Name: test_module"));
  EXPECT_THAT(output, HasSubstr("threads_per_block_limit: 1024"));
  EXPECT_THAT(output, HasSubstr("kernel_thunk (my_kernel)"));
  EXPECT_THAT(output, HasSubstr("custom_call_thunk (my_custom_call)"));
  EXPECT_THAT(output, HasSubstr("custom_kernel_thunk (my_custom_kernel)"));
  EXPECT_THAT(output, HasSubstr("kernel_thunk (nested_kernel)"));
  EXPECT_THAT(output, Not(HasSubstr("copy_thunk")));
  EXPECT_THAT(output, Not(HasSubstr("async_done_thunk")));
  EXPECT_THAT(output, Not(HasSubstr("async_start_thunk")));
  EXPECT_THAT(output, HasSubstr("kernel_thunk (async_nested_kernel)"));
}

}  // namespace
}  // namespace xla::split_proto_cli
