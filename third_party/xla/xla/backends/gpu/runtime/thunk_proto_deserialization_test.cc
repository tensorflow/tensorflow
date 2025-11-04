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

#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pointer;
using ::testing::Property;
using ::testing::WhenDynamicCastTo;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using Kind = Thunk::Kind;

constexpr absl::string_view kTestPlatformName = "TEST_PLATFORM";

TEST(ThunkProtoDeserializationTest, SequentialThunkChain) {
  constexpr ExecutionStreamId kExecutionStreamId{123};
  constexpr absl::string_view kProfileAnnotation = "profile_annotation";

  Thunk::ThunkInfo thunk_info{};
  thunk_info.execution_stream_id = kExecutionStreamId;
  thunk_info.profile_annotation = kProfileAnnotation;

  // This constructs the following thunk tree:
  // `SequentialThunk{SequentialThunk{}}`
  std::unique_ptr<Thunk> inner_thunk =
      std::make_unique<SequentialThunk>(thunk_info, ThunkSequence{});
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(inner_thunk));
  SequentialThunk outer_thunk(thunk_info, std::move(thunk_sequence));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, outer_thunk.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> new_thunk,
      DeserializeThunkProto(proto, /*buffer_allocations=*/{},
                            /*hlo_module=*/nullptr, kTestPlatformName));

  EXPECT_THAT(new_thunk.get(),
              WhenDynamicCastTo<const SequentialThunk*>(Property(
                  &SequentialThunk::thunks,
                  ElementsAre(Pointer(WhenDynamicCastTo<const SequentialThunk*>(
                      Property(&SequentialThunk::thunks, IsEmpty())))))));
}

TEST(ThunkProtoDeserializationTest, CopyThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        copy_thunk {
          source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
          destination_buffer { offset: 0 size: 256 buffer_allocation_index: 1 }
          mem_size: 256
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName));
  auto* copy_thunk = dynamic_cast<CopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, DeviceToHostCopyThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        device_to_host_copy_thunk {
          copy_thunk {
            source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
            destination_buffer {
              offset: 0
              size: 256
              buffer_allocation_index: 1
            }
            mem_size: 256
          }
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName));
  auto* copy_thunk = dynamic_cast<DeviceToHostCopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, HostToDeviceCopyThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        host_to_device_copy_thunk {
          copy_thunk {
            source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
            destination_buffer {
              offset: 0
              size: 256
              buffer_allocation_index: 1
            }
            mem_size: 256
          }
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName));
  auto* copy_thunk = dynamic_cast<HostToDeviceCopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, DeviceToDeviceCopyThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        device_to_device_copy_thunk {
          copy_thunk {
            source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
            destination_buffer {
              offset: 0
              size: 256
              buffer_allocation_index: 1
            }
            mem_size: 256
          }
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName));
  auto* copy_thunk = dynamic_cast<DeviceToDeviceCopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, WhileThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        while_thunk {
          condition_result_buffer_index {
            buffer_allocation_index: 5
            offset: 16
            size: 256
          }
          condition_thunk_sequence {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { buffer_allocation_index: 0 }
                destination_buffer { buffer_allocation_index: 1 }
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { buffer_allocation_index: 1 }
                destination_buffer { buffer_allocation_index: 2 }
              }
            }
          }
          body_thunk_sequence {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { buffer_allocation_index: 2 }
                destination_buffer { buffer_allocation_index: 3 }
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { buffer_allocation_index: 3 }
                destination_buffer { buffer_allocation_index: 4 }
              }
            }
          }
          trip_count: 10
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/3, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/4, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/5, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> athunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName));
  auto* thunk = dynamic_cast<WhileThunk*>(athunk.get());
  ASSERT_NE(thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, ConditionalThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        conditional_thunk {
          branch_index_buffer { offset: 8 size: 256 buffer_allocation_index: 5 }
          branch_thunks {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { offset: 0 size: 256 buffer_allocation_index: 0 }
                destination_buffer {
                  offset: 1
                  size: 257
                  buffer_allocation_index: 1
                }
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { offset: 2 size: 258 buffer_allocation_index: 1 }
                destination_buffer {
                  offset: 3
                  size: 259
                  buffer_allocation_index: 2
                }
              }
            }
          }
          branch_thunks {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { offset: 4 size: 260 buffer_allocation_index: 2 }
                destination_buffer {
                  offset: 5
                  size: 261
                  buffer_allocation_index: 3
                }
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { offset: 6 size: 262 buffer_allocation_index: 3 }
                destination_buffer {
                  offset: 7
                  size: 263
                  buffer_allocation_index: 4
                }
              }
            }
          }
          branch_index_is_bool: true
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/3, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/4, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/5, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> athunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName));
  auto* thunk = dynamic_cast<ConditionalThunk*>(athunk.get());
  ASSERT_NE(thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, WaitForStreamsThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { execution_stream_id: 7 }
        wait_for_streams_thunk { stream_id: 1 wait_for_stream_id: 2 }
      )pb");

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, /*buffer_allocations=*/{},
                            /*hlo_module=*/nullptr, kTestPlatformName));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, CudnnThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { execution_stream_id: 7 }
        cudnn_thunk {
          fingerprint: "fingerprint"
          args { buffer_allocation_index: 0 }
          args { buffer_allocation_index: 1 }
        }
      )pb");
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0),
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, CublasLtMatmulThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "custom-call.4" }
        cublas_lt_matmul_thunk {
          gemm_config {
            lhs_layout {
              order: ORDER_ROW_MAJOR
              num_rows: 101
              num_cols: 407
              batch_size: 1
              leading_dim_stride: 407
              dtype: F32
            }
            rhs_layout {
              order: ORDER_ROW_MAJOR
              num_rows: 407
              num_cols: 400
              batch_size: 1
              leading_dim_stride: 400
              dtype: F32
            }
            c_layout {
              order: ORDER_ROW_MAJOR
              num_rows: 101
              num_cols: 400
              batch_size: 1
              leading_dim_stride: 400
              dtype: F32
            }
            output_layout {
              order: ORDER_ROW_MAJOR
              num_rows: 101
              num_cols: 400
              batch_size: 1
              leading_dim_stride: 400
              dtype: F32
            }
            alpha_real: 1
            algorithm: -1
          }
          epilogue: EPILOGUE_DEFAULT
          canonical_hlo: "(f32[101,400]{1,0}, s8[33554432]{0}) custom-call(f32[101,407]{1,0}, f32[407,400]{1,0}), custom_call_target=\"__cublas$lt$matmul\", backend_config={\"operation_queue_id\":\"0\",\"wait_on_operation_queues\":[],\"gemm_backend_config\":{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"],\"algorithm\":\"ALG_UNSET\"},\"epilogue\":\"DEFAULT\",\"lhs_stride\":\"41107\",\"rhs_stride\":\"162800\",\"grad_x\":false,\"grad_y\":false,\"damax_output\":false},\"force_earliest_schedule\":false,\"reification_cost\":[]}"
          a { size: 164428 buffer_allocation_index: 3 }
          b { size: 651200 buffer_allocation_index: 4 }
          c { size: 161600 buffer_allocation_index: 5 }
          d { size: 161600 buffer_allocation_index: 5 }
        }
      )pb");

  std::vector<BufferAllocation> allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/3, /*size=*/164428, /*color=*/0),
      BufferAllocation(/*index=*/4, /*size=*/651200, /*color=*/0),
      BufferAllocation(/*index=*/5, /*size=*/161600, /*color=*/0),
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

XLA_FFI_DEFINE_HANDLER(kSimpleCustomCall, []() { return absl::OkStatus(); },
                       ffi::Ffi::Bind(), {ffi::Traits::kCmdBufferCompatible});

constexpr absl::string_view kSimpleCustomCallName =
    "__xla_test$$simple_custom_call";

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kSimpleCustomCallName,
                         "TEST_PLATFORM", kSimpleCustomCall);

TEST(ThunkProtoDeserializationTest, CustomCallThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { execution_stream_id: 7 }
        custom_call_thunk {
          target_name: "__xla_test$$simple_custom_call"
          operands {
            shaped_slice {
              slice { buffer_allocation_index: 0 }
              shape {
                dimensions: 42
                element_type: S32
                is_dynamic_dimension: false
              }
            }
          }
          operands {
            shaped_slice {
              slice { buffer_allocation_index: 1 }
              shape {
                dimensions: 42
                element_type: S32
                is_dynamic_dimension: false
              }
            }
          }
          results {
            shaped_slice {
              slice { buffer_allocation_index: 2 }
              shape {
                dimensions: 42
                element_type: S32
                is_dynamic_dimension: false
              }
            }
          }
          results {
            shaped_slice {
              slice { buffer_allocation_index: 3 }
              shape {
                dimensions: 42
                element_type: S32
                is_dynamic_dimension: false
              }
            }
          }
          api_version: API_VERSION_TYPED_FFI
          attributes {
            attrs {
              key: "my_attribute"
              value { scalar { i32: 42 } }
            }
          }
          called_computation: "called_computation"
        }
      )pb");
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/3, /*size=*/1024, /*color=*/0),
  };

  HloModuleConfig config;
  HloModule hlo_module("test_module", config);
  HloComputation::Builder builder("called_computation");
  // This instruction is pretty arbitrary, we just need a non-empty computation.
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(U32, {42}), "parameter"));
  hlo_module.AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, &hlo_module,
                            kTestPlatformName));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, EmptyThunkImplReturnsAnError) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { execution_stream_id: 7 }
      )pb");

  EXPECT_THAT(DeserializeThunkProto(proto, /*buffer_allocations=*/{},
                                    /*hlo_module=*/nullptr, kTestPlatformName),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::gpu
