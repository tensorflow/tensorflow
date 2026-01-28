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

#include "xla/stream_executor/kernel_spec.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

namespace stream_executor {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Field;
using ::testing::Optional;
using ::testing::Property;
using ::testing::VariantWith;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

// Creates a pointer to a CUDA kernel with a value that can be used to identify
// it later. Note that this is not a valid pointer, but that's fine as long
// as we don't try to dereference it.
void* InventPointerToCudaKernel(std::uintptr_t value) {
  return tsl::safe_reinterpret_cast<void*>(value);
}

TEST(KernelLoaderSpec, InProcessSymbol) {
  void* symbol = InventPointerToCudaKernel(0xDEADBEEF);
  auto spec = KernelLoaderSpec::CreateInProcessSymbolSpec(symbol, "kernel24",
                                                          /*arity=*/2);
  EXPECT_FALSE(spec.has_cuda_cubin_in_memory());
  EXPECT_FALSE(spec.has_cuda_ptx_in_memory());
  EXPECT_TRUE(spec.has_in_process_symbol());

  EXPECT_THAT(spec.in_process_symbol(),
              Optional(Field(&InProcessSymbol::symbol, symbol)));
  EXPECT_THAT(spec.kernel_name(), "kernel24");
}

TEST(KernelLoaderSpec, CudaCubin) {
  static constexpr std::array<uint8_t, 4> kCubinData = {0xDE, 0xAD, 0xBE, 0xEF};
  auto spec = KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      kCubinData, "kernel24", /*arity=*/2);
  EXPECT_TRUE(spec.has_cuda_cubin_in_memory());
  EXPECT_FALSE(spec.has_cuda_ptx_in_memory());
  EXPECT_FALSE(spec.has_in_process_symbol());

  EXPECT_THAT(spec.cuda_cubin_in_memory(),
              Optional(Field(&CudaCubinInMemory::cubin_bytes, kCubinData)));
  EXPECT_THAT(spec.kernel_name(), "kernel24");
}

TEST(KernelLoaderSpec, OwningCudaCubin) {
  static constexpr std::array<uint8_t, 4> kCubinData = {0xDE, 0xAD, 0xBE, 0xEF};
  auto spec = KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
      std::vector<uint8_t>{kCubinData.begin(), kCubinData.end()}, "kernel24",
      /*arity=*/2);
  EXPECT_TRUE(spec.has_cuda_cubin_in_memory());
  EXPECT_FALSE(spec.has_cuda_ptx_in_memory());
  EXPECT_FALSE(spec.has_in_process_symbol());

  EXPECT_THAT(spec.cuda_cubin_in_memory(),
              Optional(Field(&CudaCubinInMemory::cubin_bytes, kCubinData)));
  EXPECT_THAT(spec.kernel_name(), "kernel24");
}

TEST(KernelLoaderSpec, CudaPtx) {
  static constexpr absl::string_view kPtxData = "PTX DEADBEEF";
  auto spec = KernelLoaderSpec::CreateCudaPtxInMemorySpec(kPtxData, "kernel24",
                                                          /*arity=*/2);
  EXPECT_FALSE(spec.has_cuda_cubin_in_memory());
  EXPECT_TRUE(spec.has_cuda_ptx_in_memory());
  EXPECT_FALSE(spec.has_in_process_symbol());

  EXPECT_THAT(spec.cuda_ptx_in_memory(),
              Optional(Field(&CudaPtxInMemory::ptx, kPtxData)));
  EXPECT_THAT(spec.kernel_name(), "kernel24");
}

TEST(KernelLoaderSpec, OwningCudaPtx) {
  static constexpr absl::string_view kPtxData = "PTX DEADBEEF";
  auto spec = KernelLoaderSpec::CreateOwningCudaPtxInMemorySpec(
      std::string{kPtxData}, "kernel24", /*arity=*/2);
  EXPECT_FALSE(spec.has_cuda_cubin_in_memory());
  EXPECT_TRUE(spec.has_cuda_ptx_in_memory());
  EXPECT_FALSE(spec.has_in_process_symbol());

  EXPECT_THAT(spec.cuda_ptx_in_memory(),
              Optional(Field(&CudaPtxInMemory::ptx, kPtxData)));
  EXPECT_THAT(spec.kernel_name(), "kernel24");
}

TEST(KernelLoaderSpec, PtxKernelFromProto) {
  auto proto = ParseTextProtoOrDie<KernelLoaderSpecProto>(R"pb(
    ptx { data: "PTX!" }
    kernel_name: "kernel_name"
    arity: 42
  )pb");

  TF_ASSERT_OK_AND_ASSIGN(KernelLoaderSpec spec,
                          KernelLoaderSpec::FromProto(proto));
  EXPECT_THAT(spec.kernel_name(), "kernel_name");
  EXPECT_THAT(spec.arity(), 42);
  EXPECT_THAT(spec.cuda_ptx_in_memory(),
              Optional(Field(&CudaPtxInMemory::ptx, "PTX!")));
}

TEST(KernelLoaderSpec, PtxKernelToProto) {
  auto spec = KernelLoaderSpec::CreateCudaPtxInMemorySpec("PTX!", "kernel_name",
                                                          /*arity=*/42);

  EXPECT_THAT(spec.ToProto(), absl_testing::IsOkAndHolds(EqualsProto(R"pb(
                ptx { data: "PTX!" }
                kernel_name: "kernel_name"
                arity: 42
              )pb")));
}

TEST(KernelLoaderSpec, CubinKernelFromProto) {
  auto proto = ParseTextProtoOrDie<KernelLoaderSpecProto>(R"pb(
    cubin { data: "CUBIN" }
    kernel_name: "kernel_name"
    arity: 42
  )pb");

  TF_ASSERT_OK_AND_ASSIGN(KernelLoaderSpec spec,
                          KernelLoaderSpec::FromProto(proto));
  EXPECT_THAT(spec.kernel_name(), "kernel_name");
  std::array<uint8_t, 5> kCubin = {'C', 'U', 'B', 'I', 'N'};
  EXPECT_THAT(spec.arity(), 42);
  EXPECT_THAT(spec.cuda_cubin_in_memory(),
              Optional(Field(&CudaCubinInMemory::cubin_bytes, kCubin)));
}

TEST(KernelLoaderSpec, CubinKernelToProto) {
  std::array<uint8_t, 5> kCubin = {'C', 'U', 'B', 'I', 'N'};
  auto spec = KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      kCubin, "kernel_name", /*arity=*/42);

  EXPECT_THAT(spec.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
                cubin { data: "CUBIN" }
                kernel_name: "kernel_name"
                arity: 42
              )pb")));
}

TEST(KernelLoaderSpec, InProcessSymbolFromProto) {
  auto proto = ParseTextProtoOrDie<KernelLoaderSpecProto>(R"pb(
    in_process_symbol { persistent_name: "persistent_kernel_name" }
    kernel_name: "kernel_name"
    arity: 42
    kernel_args_packing_spec {
      kernel_arguments {
        relocations { kind: KIND_BITS64_ABSOLUTE argument_index: 0 offset: 0 }
        data: "\x00\x00\x00\x00\x00\x00\x00\x00"
      }
      kernel_arguments { data: "\x34\x12\x00\x00" }
    }
  )pb");

  const auto symbol_resolver = [](absl::string_view name) {
    return InventPointerToCudaKernel(0x1234567890);
  };

  TF_ASSERT_OK_AND_ASSIGN(KernelLoaderSpec spec,
                          KernelLoaderSpec::FromProto(proto, symbol_resolver));
  EXPECT_EQ(spec.kernel_name(), "kernel_name");
  EXPECT_EQ(spec.arity(), 42);
  EXPECT_THAT(spec.in_process_symbol(),
              Optional(Field(&InProcessSymbol::symbol,
                             InventPointerToCudaKernel(0x1234567890))));
  EXPECT_THAT(spec.in_process_symbol(),
              Optional(Field(&InProcessSymbol::persistent_name,
                             "persistent_kernel_name")));

  const auto kReferenceKernelArgsPackingSpecProto =
      R"pb(
    kernel_arguments {
      relocations { kind: KIND_BITS64_ABSOLUTE argument_index: 0 offset: 0 }
      data: "\x00\x00\x00\x00\x00\x00\x00\x00"
    }
    kernel_arguments { data: "\x34\x12\x00\x00" }
      )pb";
  EXPECT_THAT(
      spec.kernel_args_packing(),
      VariantWith<KernelArgsPackingSpec>(Property(
          &KernelArgsPackingSpec::ToProto,
          IsOkAndHolds(EqualsProto(kReferenceKernelArgsPackingSpecProto)))));

  // If the symbol resolver is not provided, the spec cannot be deserialized.
  EXPECT_THAT(KernelLoaderSpec::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KernelLoaderSpec, InProcessSymbolToProto) {
  auto non_serializable_spec = KernelLoaderSpec::CreateInProcessSymbolSpec(
      /*symbol=*/nullptr, "kernel_name", 42);

  // InProcessSymbol specs without a persistent name cannot be serialized.
  EXPECT_THAT(non_serializable_spec.ToProto(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto serializable_spec =
      KernelLoaderSpec::CreateSerializableInProcessSymbolSpec(
          "persistent_kernel_name", nullptr, "kernel_name", 42);
  EXPECT_THAT(serializable_spec.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
                in_process_symbol { persistent_name: "persistent_kernel_name" }
                kernel_name: "kernel_name"
                arity: 42
              )pb")));
}

TEST(kernelLoaderSpec, StoresKernelArgsPackingSpec) {
  auto kernel_args_packing_spec_proto =
      ParseTextProtoOrDie<KernelArgsPackingSpecProto>(
          R"pb(
            kernel_arguments {
              relocations {
                kind: KIND_BITS64_ABSOLUTE
                argument_index: 0
                offset: 0
              }
              data: "\x00\x00\x00\x00\x00\x00\x00\x00"
            }
            kernel_arguments { data: "\x34\x12\x00\x00" }
          )pb");

  TF_ASSERT_OK_AND_ASSIGN(
      KernelArgsPackingSpec kernel_args_packing_spec,
      KernelArgsPackingSpec::FromProto(kernel_args_packing_spec_proto));

  auto spec = KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
      std::vector<uint8_t>{'C', 'U', 'B', 'I', 'N'}, "kernel_name",
      /*arity=*/42, std::move(kernel_args_packing_spec));

  EXPECT_THAT(spec.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
                cubin { data: "CUBIN" }
                kernel_name: "kernel_name"
                arity: 42
                kernel_args_packing_spec {
                  kernel_arguments {
                    relocations {
                      kind: KIND_BITS64_ABSOLUTE
                      argument_index: 0
                      offset: 0
                    }
                    data: "\x00\x00\x00\x00\x00\x00\x00\x00"
                  }
                  kernel_arguments { data: "\x34\x12\x00\x00" }
                }
              )pb")));
}

}  // namespace
}  // namespace stream_executor
