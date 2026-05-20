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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/device_to_host_copy_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/host_execute_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/host_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
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
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::StatusOr<std::unique_ptr<Thunk>> DeserializeThunkProto(
    const ThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const HloModule* absl_nullable hlo_module, absl::string_view platform_name,
    const se::GpuComputeCapability& gpu_compute_capability,
    const std::optional<stream_executor::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver = std::nullopt) {
  ThunkSequenceProto thunk_sequence_proto;
  *thunk_sequence_proto.add_thunks() = thunk_proto;
  TF_ASSIGN_OR_RETURN(
      ThunkSequence sequence,
      DeserializeThunkSequenceProto(thunk_sequence_proto, buffer_allocations,
                                    hlo_module, platform_name,
                                    gpu_compute_capability, symbol_resolver));
  return std::move(sequence.front());
}

using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::Pointer;
using ::testing::Property;
using ::testing::WhenDynamicCastTo;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using Kind = Thunk::Kind;

constexpr absl::string_view kTestPlatformName = "TEST_PLATFORM";

TEST(ThunkProtoDeserializationTest, SequentialThunkChain) {
  constexpr absl::string_view kProfileAnnotation = "profile_annotation";

  Thunk::ThunkInfo thunk_info{};
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
                            /*hlo_module=*/nullptr, kTestPlatformName,
                            se::GpuComputeCapability()));

  EXPECT_THAT(new_thunk.get(),
              WhenDynamicCastTo<const SequentialThunk*>(Property(
                  &SequentialThunk::thunks,
                  ElementsAre(Pointer(WhenDynamicCastTo<const SequentialThunk*>(
                      Property(&SequentialThunk::thunks, IsEmpty())))))));
}

TEST(ThunkProtoDeserializationTest, CopyThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        copy_thunk {
          source_buffer {
            slice { offset: 128 size: 384 buffer_allocation_index: 0 }
            shape {
              dimensions: 64
              element_type: S32
              is_dynamic_dimension: false
              layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            }
          }
          destination_buffer {
            slice { offset: 0 size: 256 buffer_allocation_index: 1 }
            shape {
              dimensions: 64
              element_type: S32
              is_dynamic_dimension: false
              layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            }
          }
          mem_size: 256
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName, se::GpuComputeCapability()));
  auto* copy_thunk = dynamic_cast<CopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, DeviceToHostCopyThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        device_to_host_copy_thunk {
          copy_thunk {
            source_buffer {
              slice { offset: 128 size: 384 buffer_allocation_index: 0 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            destination_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 1 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
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
                            kTestPlatformName, se::GpuComputeCapability()));
  auto* copy_thunk = dynamic_cast<DeviceToHostCopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, HostToDeviceCopyThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        host_to_device_copy_thunk {
          copy_thunk {
            source_buffer {
              slice { offset: 128 size: 384 buffer_allocation_index: 0 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            destination_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 1 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
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
                            kTestPlatformName, se::GpuComputeCapability()));
  auto* copy_thunk = dynamic_cast<HostToDeviceCopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, DeviceToDeviceCopyThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        device_to_device_copy_thunk {
          copy_thunk {
            source_buffer {
              slice { offset: 128 size: 384 buffer_allocation_index: 0 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            destination_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 1 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
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
                            kTestPlatformName, se::GpuComputeCapability()));
  auto* copy_thunk = dynamic_cast<DeviceToDeviceCopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, WhileThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        while_thunk {
          condition_result_buffer_index {
            buffer_allocation_index: 5
            offset: 16
            size: 256
          }
          condition_thunk_sequence {
            thunks {
              thunk_info { profile_annotation: "profile_annotation" }
              copy_thunk {
                source_buffer {
                  slice { offset: 128 size: 384 buffer_allocation_index: 0 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
                destination_buffer {
                  slice { offset: 0 size: 256 buffer_allocation_index: 1 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
              }
            }
          }
          body_thunk_sequence {
            thunks {
              thunk_info { profile_annotation: "profile_annotation" }
              copy_thunk {
                source_buffer {
                  slice { offset: 128 size: 384 buffer_allocation_index: 2 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
                destination_buffer {
                  slice { offset: 0 size: 256 buffer_allocation_index: 3 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
              }
            }
            thunks {
              thunk_info { profile_annotation: "profile_annotation" }
              copy_thunk {
                source_buffer {
                  slice { offset: 128 size: 384 buffer_allocation_index: 3 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
                destination_buffer {
                  slice { offset: 0 size: 256 buffer_allocation_index: 4 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
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
                            kTestPlatformName, se::GpuComputeCapability()));
  auto* thunk = dynamic_cast<WhileThunk*>(athunk.get());
  ASSERT_NE(thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, ConditionalThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        conditional_thunk {
          branch_index_buffer {
            slice { offset: 8 size: 1 buffer_allocation_index: 5 }
            shape { element_type: PRED }
          }
          branch_thunks {
            thunks {
              thunk_info { profile_annotation: "profile_annotation" }
              copy_thunk {
                source_buffer {
                  slice { offset: 0 size: 256 buffer_allocation_index: 0 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
                destination_buffer {
                  slice { offset: 1 size: 257 buffer_allocation_index: 1 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
              }
            }
            thunks {
              thunk_info { profile_annotation: "profile_annotation" }
              copy_thunk {
                source_buffer {
                  slice { offset: 2 size: 258 buffer_allocation_index: 1 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
                destination_buffer {
                  slice { offset: 3 size: 259 buffer_allocation_index: 2 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
              }
            }
          }
          branch_thunks {
            thunks {
              thunk_info { profile_annotation: "profile_annotation" }
              copy_thunk {
                source_buffer {
                  slice { offset: 4 size: 260 buffer_allocation_index: 3 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
                destination_buffer {
                  slice { offset: 5 size: 261 buffer_allocation_index: 3 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
              }
            }
            thunks {
              thunk_info { profile_annotation: "profile_annotation" }
              copy_thunk {
                source_buffer {
                  slice { offset: 6 size: 262 buffer_allocation_index: 3 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
                destination_buffer {
                  slice { offset: 7 size: 263 buffer_allocation_index: 4 }
                  shape {
                    dimensions: 64
                    element_type: S32
                    is_dynamic_dimension: false
                    layout {
                      minor_to_major: 0
                      tail_padding_alignment_in_elements: 1
                    }
                  }
                }
              }
            }
          }
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
                            kTestPlatformName, se::GpuComputeCapability()));
  auto* thunk = dynamic_cast<ConditionalThunk*>(athunk.get());
  ASSERT_NE(thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, CudnnThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {}
        cudnn_thunk {
          fingerprint: "fingerprint"
          args {
            slice { buffer_allocation_index: 0 }
            shape { element_type: U8 }
          }
          args {
            slice { buffer_allocation_index: 1 }
            shape { element_type: U8 }
          }
        }
      )pb");
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0),
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName, se::GpuComputeCapability()));

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
          canonical_hlo: "(f32[101,400]{1,0}, s8[33554432]{0}) custom-call(f32[101,407]{1,0}, f32[407,400]{1,0}), custom_call_target=\"__cublas$lt$matmul\", backend_config={\"operation_queue_id\":\"0\",\"gemm_backend_config\":{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"],\"algorithm\":\"ALG_UNSET\"},\"epilogue\":\"DEFAULT\",\"lhs_stride\":\"41107\",\"rhs_stride\":\"162800\",\"grad_x\":false,\"grad_y\":false,\"damax_output\":false},\"force_earliest_schedule\":false,\"reification_cost\":[]}"
          a {
            slice { size: 164428 buffer_allocation_index: 3 }
            shape {
              element_type: F32
              dimensions: 101
              dimensions: 407
              is_dynamic_dimension: false
              is_dynamic_dimension: false
            }
          }
          b {
            slice { size: 651200 buffer_allocation_index: 4 }
            shape {
              element_type: F32
              dimensions: 407
              dimensions: 400
              is_dynamic_dimension: false
              is_dynamic_dimension: false
            }
          }
          c {
            slice { size: 161600 buffer_allocation_index: 5 }
            shape {
              element_type: F32
              dimensions: 101
              dimensions: 400
              is_dynamic_dimension: false
              is_dynamic_dimension: false
            }
          }
          d {
            slice { size: 161600 buffer_allocation_index: 5 }
            shape {
              element_type: F32
              dimensions: 101
              dimensions: 400
              is_dynamic_dimension: false
              is_dynamic_dimension: false
            }
          }
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
                            kTestPlatformName, se::GpuComputeCapability()));

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
        thunk_info {}
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
          execution_state {}
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
                            kTestPlatformName, se::GpuComputeCapability()));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, CublasLtGroupedMatmulThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "custom-call.4" }
        cublas_lt_matmul_thunk {
          grouped_gemm_config {
            m: 101
            n: 400
            k: 407
            batch_count: 1
            group_count: 1
            lhs_leading_dim_stride: 407
            rhs_leading_dim_stride: 400
            c_leading_dim_stride: 400
            output_leading_dim_stride: 400
            trans_a: BLAS_TRANSPOSE
            trans_b: BLAS_TRANSPOSE
            must_swap_operands: True
            alpha_real: 1
            type_a: F32
            type_b: F32
            type_c: F32
            type_d: F32
            stride_ragged_dim: 407
            stride_group_dim: 400
            output_stride_ragged_dim: 400
            ragged_mode: RAGGED_NON_CONTRACTING
          }
          epilogue: EPILOGUE_DEFAULT
          canonical_hlo: "(f32[101,400]{1,0}, s8[33554432]{0}) custom-call(f32[101,407]{1,0}, f32[2, 407,400]{1,0}, s32[2]{0}), custom_call_target=\"__cublas$lt$groupedMatmul backend_config={\"operation_queue_id\":\"0\", \"force_earliest_schedule\":false,\"reification_cost\":[], \"device_type\":\"DEVICE_TYPE_INVALID\", \"grouped_gemm_backend_config\":{\"gemm_backend_config\":{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"epilogue\":\"DEFAULT\",\"grad_x\":false,\"grad_y\":false,\"damax_output\":false},\"ragged_dot_dimension_numbers\":{\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"lhs_ragged_dimensions\":[\"0\"],\"rhs_group_dimensions\":[\"0\"]}}})"
          a {
            slice { size: 164428 buffer_allocation_index: 3 }
            shape { element_type: F32 }
          }
          b {
            slice { size: 1302400 buffer_allocation_index: 4 }
            shape { element_type: F32 }
          }
          c {
            slice { size: 161600 buffer_allocation_index: 5 }
            shape { element_type: F32 }
          }
          d {
            slice { size: 161600 buffer_allocation_index: 5 }
            shape { element_type: F32 }
          }
          group_sizes {
            slice { size: 8 buffer_allocation_index: 6 }
            shape { element_type: S32 }
          }
        }
      )pb");

  std::vector<BufferAllocation> allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/3, /*size=*/164428, /*color=*/0),
      BufferAllocation(/*index=*/4, /*size=*/651200, /*color=*/0),
      BufferAllocation(/*index=*/5, /*size=*/161600, /*color=*/0),
      BufferAllocation(/*index=*/6, /*size=*/8, /*color=*/0),
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName, se::GpuComputeCapability()));
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, EmptyThunkImplReturnsAnError) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {}
      )pb");

  EXPECT_THAT(DeserializeThunkProto(proto, /*buffer_allocations=*/{},
                                    /*hlo_module=*/nullptr, kTestPlatformName,
                                    se::GpuComputeCapability()),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ThunkProtoDeserializationTest, HostSendRecvThunksRoundTrip) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {}
        sequential_thunk {
          thunks {
            thunk_info {}
            host_send_thunk {
              shape {
                element_type: F32
                dimensions: [ 10 ]
                is_dynamic_dimension: false
              }
              buffer { buffer_allocation_index: 0 }
              channel_id: 123
              async_events_unique_id: 1
            }
          }
          thunks {
            thunk_info {}
            host_send_done_thunk { channel_id: 123 async_events_unique_id: 1 }
          }
          thunks {
            thunk_info {}
            host_recv_thunk {
              shape {
                element_type: F32
                dimensions: [ 10 ]
                is_dynamic_dimension: false

              }
              buffer { buffer_allocation_index: 0 }
              channel_id: 456
              async_events_unique_id: 2
            }
          }
          thunks {
            thunk_info {}
            host_recv_done_thunk { channel_id: 456 async_events_unique_id: 2 }
          }
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations,
                            /*hlo_module=*/nullptr, kTestPlatformName,
                            se::GpuComputeCapability()));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  const auto* sequential_thunk = dynamic_cast<SequentialThunk*>(thunk.get());
  ASSERT_NE(sequential_thunk, nullptr);
  ASSERT_EQ(sequential_thunk->thunks().size(), 4);

  const auto* send_thunk =
      dynamic_cast<HostSendThunk*>(sequential_thunk->thunks()[0].get());
  ASSERT_NE(send_thunk, nullptr);

  const auto* send_done_thunk =
      dynamic_cast<HostSendDoneThunk*>(sequential_thunk->thunks()[1].get());
  ASSERT_NE(send_done_thunk, nullptr);

  const auto* recv_thunk =
      dynamic_cast<HostRecvThunk*>(sequential_thunk->thunks()[2].get());
  ASSERT_NE(recv_thunk, nullptr);

  const auto* recv_done_thunk =
      dynamic_cast<HostRecvDoneThunk*>(sequential_thunk->thunks()[3].get());
  ASSERT_NE(recv_done_thunk, nullptr);

  EXPECT_TRUE(send_thunk->GetAsyncEventsUniqueId().has_value());
  EXPECT_TRUE(send_done_thunk->GetAsyncEventsUniqueId().has_value());
  EXPECT_EQ(send_thunk->GetAsyncEventsUniqueId(),
            send_done_thunk->GetAsyncEventsUniqueId());

  EXPECT_TRUE(recv_thunk->GetAsyncEventsUniqueId().has_value());
  EXPECT_TRUE(recv_done_thunk->GetAsyncEventsUniqueId().has_value());
  EXPECT_EQ(recv_thunk->GetAsyncEventsUniqueId(),
            recv_done_thunk->GetAsyncEventsUniqueId());

  // The unique id is regenerated on deserialization. Overwrite it with the
  // original value for the purpose of the roundtrip test.
  round_trip_proto.mutable_sequential_thunk()
      ->mutable_thunks(0)
      ->mutable_host_send_thunk()
      ->set_async_events_unique_id(1);
  round_trip_proto.mutable_sequential_thunk()
      ->mutable_thunks(1)
      ->mutable_host_send_done_thunk()
      ->set_async_events_unique_id(1);
  round_trip_proto.mutable_sequential_thunk()
      ->mutable_thunks(2)
      ->mutable_host_recv_thunk()
      ->set_async_events_unique_id(2);
  round_trip_proto.mutable_sequential_thunk()
      ->mutable_thunks(3)
      ->mutable_host_recv_done_thunk()
      ->set_async_events_unique_id(2);
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, HostExecuteThunksRoundTrip) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {}
        sequential_thunk {
          thunks {
            thunk_info {}
            host_execute_start_thunk {
              executable_proto { executable_type: EXECUTABLE_TYPE_NANORT }
              async_events_unique_id: 123
            }
          }
          thunks {
            thunk_info {}
            host_execute_done_thunk { async_events_unique_id: 123 }
          }
        }
      )pb");

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, /*buffer_allocations=*/{},
                            /*hlo_module=*/nullptr, kTestPlatformName,
                            se::GpuComputeCapability()));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  // Check that start and done thunks share the same async event id.
  const auto* sequential_thunk = dynamic_cast<SequentialThunk*>(thunk.get());
  ASSERT_NE(sequential_thunk, nullptr);
  ASSERT_EQ(sequential_thunk->thunks().size(), 2);

  const auto* start_thunk =
      dynamic_cast<HostExecuteStartThunk*>(sequential_thunk->thunks()[0].get());
  ASSERT_NE(start_thunk, nullptr);

  const auto* done_thunk =
      dynamic_cast<HostExecuteDoneThunk*>(sequential_thunk->thunks()[1].get());
  ASSERT_NE(done_thunk, nullptr);

  EXPECT_TRUE(start_thunk->GetAsyncEventsUniqueId().has_value());
  EXPECT_TRUE(done_thunk->GetAsyncEventsUniqueId().has_value());
  EXPECT_EQ(start_thunk->GetAsyncEventsUniqueId(),
            done_thunk->GetAsyncEventsUniqueId());

  // The unique id is regenerated on deserialization. Overwrite it with the
  // original value for the purpose of the roundtrip test.
  round_trip_proto.mutable_sequential_thunk()
      ->mutable_thunks(0)
      ->mutable_host_execute_start_thunk()
      ->set_async_events_unique_id(123);
  round_trip_proto.mutable_sequential_thunk()
      ->mutable_thunks(1)
      ->mutable_host_execute_done_thunk()
      ->set_async_events_unique_id(123);
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, CustomKernelThunkRoundTrip) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {}
        custom_kernel_thunk {
          custom_kernel {
            name: "test_kernel"
            kernel_spec {
              ptx { data: "PTX" }
              arity: 1
            }
            block_dims { coordinates { x: 1, y: 1, z: 1 } }
            thread_dims { coordinates { x: 1, y: 1, z: 1 } }
            shared_memory_bytes: 42
          }
          args {
            slice { buffer_allocation_index: 0 }
            shape {
              element_type: F32
              dimensions: 22
              is_dynamic_dimension: false
            }
          }
          written: true
          tma_metadata {}
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName, se::GpuComputeCapability()));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

// A test symbol that we can resolve to.
void test_kernel(void* args) {}

TEST(ThunkProtoDeserializationTest, CustomKernelThunkSymbolResolvingWorks) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {}
        custom_kernel_thunk {
          custom_kernel {
            name: "test_kernel"
            kernel_spec {
              in_process_symbol { persistent_name: "test_kernel" }
              arity: 1
            }
            block_dims { coordinates { x: 1, y: 1, z: 1 } }
            thread_dims { coordinates { x: 1, y: 1, z: 1 } }
            shared_memory_bytes: 42
          }
          args {
            slice { buffer_allocation_index: 0 }
            shape {
              element_type: F32
              dimensions: 22
              is_dynamic_dimension: false
            }
          }
          written: true
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0)};

  auto symbol_resolver =
      [&](absl::string_view persistent_name) -> absl::StatusOr<void*> {
    if (persistent_name == "test_kernel") {
      return tsl::safe_reinterpret_cast<void*>(&test_kernel);
    }
    return absl::NotFoundError("Symbol not found");
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(
          proto, buffer_allocations, /*hlo_module=*/nullptr, kTestPlatformName,
          stream_executor::GpuComputeCapability(), symbol_resolver));

  auto custom_kernel_thunk = dynamic_cast<CustomKernelThunk*>(thunk.get());
  ASSERT_NE(custom_kernel_thunk, nullptr);
  EXPECT_THAT(
      custom_kernel_thunk->custom_kernel().kernel_spec().in_process_symbol(),
      Optional(Field(&stream_executor::InProcessSymbol::symbol,
                     tsl::safe_reinterpret_cast<void*>(&test_kernel))));
}

TEST(ThunkProtoDeserializationTest, HostToDeviceCopyThunkRoundTrip) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {}
        host_to_device_copy_thunk {
          copy_thunk {
            source_buffer {
              slice { offset: 0 size: 1024 buffer_allocation_index: 0 }
              shape {
                dimensions: 256
                element_type: F32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            destination_buffer {
              slice { offset: 0 size: 1024 buffer_allocation_index: 1 }
              shape {
                dimensions: 256
                element_type: F32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            mem_size: 1024
          }
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations,
                            /*hlo_module=*/nullptr, kTestPlatformName,
                            se::GpuComputeCapability()));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  const auto* h2d_thunk = dynamic_cast<HostToDeviceCopyThunk*>(thunk.get());
  ASSERT_NE(h2d_thunk, nullptr);
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, AsyncStartAndDoneThunk) {
  // Serialize an AsyncStartThunk with an empty nested thunk sequence and a
  // corresponding AsyncDoneThunk, then deserialize them and verify the
  // round-trip.
  Thunk::ThunkInfo start_info;
  start_info.profile_annotation = "async_start";

  AsyncStartThunk start_thunk(start_info, ComputationStreamId(0),
                              ThunkSequence{});

  AsyncDoneThunk done_thunk(Thunk::ThunkInfo(), start_thunk.async_execution());

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto start_proto, start_thunk.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto done_proto, done_thunk.ToProto());

  // Deserialize both thunks together so the AsyncExecutionMap connects them.
  ThunkSequenceProto thunk_protos;
  *thunk_protos.add_thunks() = start_proto;
  *thunk_protos.add_thunks() = done_proto;
  TF_ASSERT_OK_AND_ASSIGN(
      ThunkSequence sequence,
      DeserializeThunkSequenceProto(thunk_protos, /*buffer_allocations=*/{},
                                    /*hlo_module=*/nullptr, kTestPlatformName,
                                    se::GpuComputeCapability()));

  ASSERT_EQ(sequence.size(), 2);
  EXPECT_EQ(sequence[0]->kind(), Kind::kAsyncStart);
  EXPECT_EQ(sequence[1]->kind(), Kind::kAsyncDone);

  // Both thunks share the same AsyncExecution instance.
  auto* deserialized_start = dynamic_cast<AsyncStartThunk*>(sequence[0].get());
  auto* deserialized_done = dynamic_cast<AsyncDoneThunk*>(sequence[1].get());
  ASSERT_NE(deserialized_start, nullptr);
  ASSERT_NE(deserialized_done, nullptr);
  EXPECT_EQ(deserialized_start->async_execution().get(),
            deserialized_done->async_execution().get());

  // Verify the round-trip by re-serializing and comparing protos. The
  // async_execution_id is derived from the shared_ptr address, so it changes
  // across serialization boundaries. Overwrite it to match the original.
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_start,
                          deserialized_start->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_done,
                          deserialized_done->ToProto());

  uint64_t new_id = round_trip_start.async_start_thunk().async_execution_id();
  start_proto.mutable_async_start_thunk()->set_async_execution_id(new_id);
  done_proto.mutable_async_done_thunk()->set_async_execution_id(new_id);

  EXPECT_THAT(round_trip_start, EqualsProto(start_proto));
  EXPECT_THAT(round_trip_done, EqualsProto(done_proto));
}

TEST(ThunkProtoDeserializationTest, SendThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        send_thunk {
          buffer {
            element_count: 64
            source_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 0 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            destination_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 1 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
          }
          collective_config {
            operand_element_type: S32
            replica_groups { replica_ids: 0 replica_ids: 1 }
            group_mode: COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          }
          source_target_pairs { source: 0 target: 1 }
          instruction_name: "send"
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName, se::GpuComputeCapability()));
  auto* send_thunk = dynamic_cast<SendThunk*>(thunk.get());
  ASSERT_NE(send_thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, send_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, RecvThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        recv_thunk {
          buffer {
            element_count: 64
            source_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 0 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            destination_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 1 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
          }
          collective_config {
            operand_element_type: S32
            replica_groups { replica_ids: 0 replica_ids: 1 }
            group_mode: COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          }
          source_target_pairs { source: 0 target: 1 }
          instruction_name: "recv"
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName, se::GpuComputeCapability()));
  auto* recv_thunk = dynamic_cast<RecvThunk*>(thunk.get());
  ASSERT_NE(recv_thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, recv_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, CollectiveKernelThunk) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        collective_kernel_thunk {
          collective_config {
            operand_element_type: S32
            replica_groups { replica_ids: 0 replica_ids: 1 }
            group_mode: COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          }
          reduction_kind: REDUCTION_KIND_SUM
          is_async: false
          buffers {
            element_count: 64
            source_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 0 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            destination_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 1 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
          }
          collective_kernel_enabled: true
          kernel_name: "my_kernel"
          launch_dimensions {
            block_counts { coordinates { x: 1 y: 2 z: 3 } }
            thread_counts_per_block { coordinates { x: 4 y: 5 z: 6 } }
          }
          shmem_bytes: 1024
          is_multimem_enabled: false
          cubin: "my_cubin"
          use_pdl: false
        }
      )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      DeserializeThunkProto(proto, buffer_allocations, /*hlo_module=*/nullptr,
                            kTestPlatformName, se::GpuComputeCapability()));
  auto* kernel_thunk = dynamic_cast<CollectiveKernelThunk*>(thunk.get());
  ASSERT_NE(kernel_thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, kernel_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace

}  // namespace xla::gpu
