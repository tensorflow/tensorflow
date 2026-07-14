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

#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.pb.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {
using ::testing::Field;
using ::testing::Optional;
using tsl::proto_testing::ParseTextProtoOrDie;

void SomeKernel(int* x) { *x = 42; }

TEST(CustomKernelTest, ToProto) {
  CustomKernel custom_kernel(
      "kernel_name",
      stream_executor::KernelLoaderSpec::CreateSerializableInProcessSymbolSpec(
          "persistent_kernel_name",
          /*symbol=*/absl::bit_cast<void*>(&SomeKernel), "kernel_name",
          /*arity=*/42),
      stream_executor::BlockDim(1, 2, 3), stream_executor::ThreadDim(4, 5, 6),
      /*shared_memory_bytes=*/7);
  TF_ASSERT_OK_AND_ASSIGN(CustomKernelProto proto, custom_kernel.ToProto());

  EXPECT_THAT(
      proto, tsl::proto_testing::EqualsProto(R"pb(
        name: "kernel_name"
        kernel_spec {
          in_process_symbol { persistent_name: "persistent_kernel_name" }
          kernel_name: "kernel_name"
          arity: 42
        }
        block_dims { coordinates { x: 1 y: 2 z: 3 } }
        thread_dims { coordinates { x: 4 y: 5 z: 6 } }
        shared_memory_bytes: 7
      )pb"));
}

TEST(CustomKernelTest, ToProtoWithClusterDims) {
  CustomKernel custom_kernel(
      "kernel_name",
      stream_executor::KernelLoaderSpec::CreateSerializableInProcessSymbolSpec(
          "persistent_kernel_name",
          /*symbol=*/absl::bit_cast<void*>(&SomeKernel), "kernel_name_in_spec",
          /*arity=*/42),
      stream_executor::BlockDim(1, 2, 3), stream_executor::ThreadDim(4, 5, 6),
      stream_executor::ClusterDim(7, 8, 9),
      /*shared_memory_bytes=*/10);
  TF_ASSERT_OK_AND_ASSIGN(CustomKernelProto proto, custom_kernel.ToProto());

  EXPECT_THAT(
      proto, tsl::proto_testing::EqualsProto(R"pb(
        name: "kernel_name"
        kernel_spec {
          in_process_symbol { persistent_name: "persistent_kernel_name" }
          kernel_name: "kernel_name_in_spec"
          arity: 42
        }
        block_dims { coordinates { x: 1 y: 2 z: 3 } }
        thread_dims { coordinates { x: 4 y: 5 z: 6 } }
        cluster_dim { coordinates { x: 7 y: 8 z: 9 } }
        shared_memory_bytes: 10
      )pb"));
}

absl::StatusOr<void*> StaticSymbolResolver(absl::string_view persistent_name) {
  // Resolves a symbol to the address of SomeKernel - no matter what the
  // persistent name is.
  return absl::bit_cast<void*>(&SomeKernel);
}

TEST(CustomKernelTest, FromProto) {
  auto proto = ParseTextProtoOrDie<CustomKernelProto>(R"pb(
    name: "kernel_name"
    kernel_spec {
      in_process_symbol { persistent_name: "persistent_kernel_name" }
      kernel_name: "kernel_name_in_spec"
      arity: 42
    }
    block_dims { coordinates { x: 1 y: 2 z: 3 } }
    thread_dims { coordinates { x: 4 y: 5 z: 6 } }
    shared_memory_bytes: 7
  )pb");
  TF_ASSERT_OK_AND_ASSIGN(CustomKernel custom_kernel,
                          CustomKernel::FromProto(proto, StaticSymbolResolver));
  EXPECT_EQ(custom_kernel.name(), "kernel_name");
  EXPECT_EQ(custom_kernel.kernel_spec().kernel_name(), "kernel_name_in_spec");
  EXPECT_EQ(custom_kernel.kernel_spec().arity(), 42);
  EXPECT_THAT(custom_kernel.kernel_spec().in_process_symbol(),
              Optional(Field(&stream_executor::InProcessSymbol::symbol,
                             absl::bit_cast<void*>(&SomeKernel))));
  EXPECT_THAT(custom_kernel.kernel_spec().in_process_symbol(),
              Optional(Field(&stream_executor::InProcessSymbol::persistent_name,
                             "persistent_kernel_name")));
  EXPECT_EQ(custom_kernel.block_dims(), stream_executor::BlockDim(1, 2, 3));
  EXPECT_EQ(custom_kernel.thread_dims(), stream_executor::ThreadDim(4, 5, 6));
  EXPECT_EQ(custom_kernel.cluster_dims(), std::nullopt);
  EXPECT_EQ(custom_kernel.shared_memory_bytes(), 7);
}

TEST(CustomKernelTest, FromProtoWithClusterDims) {
  auto proto = ParseTextProtoOrDie<CustomKernelProto>(R"pb(
    name: "kernel_name"
    kernel_spec {
      in_process_symbol { persistent_name: "persistent_kernel_name" }
      kernel_name: "kernel_name_in_spec"
      arity: 42
    }
    block_dims { coordinates { x: 1 y: 2 z: 3 } }
    thread_dims { coordinates { x: 4 y: 5 z: 6 } }
    cluster_dim { coordinates { x: 7 y: 8 z: 9 } }
    shared_memory_bytes: 10
  )pb");
  TF_ASSERT_OK_AND_ASSIGN(CustomKernel custom_kernel,
                          CustomKernel::FromProto(proto, StaticSymbolResolver));
  EXPECT_EQ(custom_kernel.name(), "kernel_name");
  EXPECT_EQ(custom_kernel.kernel_spec().kernel_name(), "kernel_name_in_spec");
  EXPECT_EQ(custom_kernel.kernel_spec().arity(), 42);
  EXPECT_THAT(custom_kernel.kernel_spec().in_process_symbol(),
              Optional(Field(&stream_executor::InProcessSymbol::symbol,
                             absl::bit_cast<void*>(&SomeKernel))));
  EXPECT_THAT(custom_kernel.kernel_spec().in_process_symbol(),
              Optional(Field(&stream_executor::InProcessSymbol::persistent_name,
                             "persistent_kernel_name")));
  EXPECT_EQ(custom_kernel.block_dims(), stream_executor::BlockDim(1, 2, 3));
  EXPECT_EQ(custom_kernel.thread_dims(), stream_executor::ThreadDim(4, 5, 6));
  EXPECT_EQ(custom_kernel.cluster_dims(), stream_executor::ClusterDim(7, 8, 9));
  EXPECT_EQ(custom_kernel.shared_memory_bytes(), 10);
}

}  // namespace
}  // namespace xla::gpu
