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

#include "xla/stream_executor/kernel_args_packing_spec.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

namespace stream_executor {

// Struct for testing custom kernel argument packing.
struct CustomData {
  int32_t value;
};

template <>
struct KernelArgPacking<CustomData> {
  using Type = int32_t;
  static int32_t Pack(CustomData data) { return data.value + 1; }
};

namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::SizeIs;
using tsl::proto_testing::EqualsProto;
using tsl::proto_testing::ParseTextProtoOrDie;

// This function creates a `DeviceAddressBase` with an opaque pointer that
// contains the given value. The size of the device memory is set to 0 since
// it's unused.
// Note that this device pointer is not a valid pointer to device memory, it
// is only used for testing and can't be dereferenced.
DeviceAddressBase MakeDevicePointer(uint32_t value) {
  // To construct a pointer that works both on 32bit and 64bit platforms and
  // does not invoke undefined behaviour, we first cast our integer to uintptr_t
  // and then cast it to void*.
  return DeviceAddressBase(
      tsl::safe_reinterpret_cast<void*>(static_cast<uintptr_t>(value)),
      /*size=*/0);
}

TEST(KernelArgPackingSpecTest, WriteArgumentAddress) {
  KernelArgPackingSpec first_arg;
  first_arg.WriteArgumentAddress(/*argument_index=*/2);

  // We fail if not enough arguments are provided.Since we are referencing
  // argument #2, we will need to provide 3 arguments.
  EXPECT_THAT(
      first_arg.BuildArgument({MakeDevicePointer(0), MakeDevicePointer(0)}),
      StatusIs(absl::StatusCode::kInvalidArgument));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<char> first_arg_storage,
      first_arg.BuildArgument({MakeDevicePointer(0), MakeDevicePointer(0),
                               MakeDevicePointer(0xff42)}));
  EXPECT_THAT(first_arg_storage,
              ElementsAre(0x42, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00));
}

TEST(KernelArgPackingSpecTest, WriteMultipleArgumentAddresses) {
  KernelArgPackingSpec first_arg;
  first_arg.WriteArgumentAddress(/*argument_index=*/0);
  first_arg.WriteArgumentAddress(/*argument_index=*/1);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<char> first_arg_storage,
      first_arg.BuildArgument(
          {MakeDevicePointer(0xff42), MakeDevicePointer(0xaabbccdd)}));
  EXPECT_THAT(first_arg_storage,
              ElementsAre(0x42, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xdd,
                          0xcc, 0xbb, 0xaa, 0x00, 0x00, 0x00, 0x00));
}

TEST(KernelArgPackingSpecTest, WriteConstant) {
  KernelArgPackingSpec first_arg;
  first_arg.WriteConstant(static_cast<uint32_t>(0x1348));
  first_arg.WriteConstant(static_cast<uint64_t>(0x2389));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<char> first_arg_storage,
                          first_arg.BuildArgument(/*args=*/{}));

  // KernelArgPackingSpec::WriteConstant doesn't take endianness into
  // account, so this assertion will fail for big endian architectures - which
  // we don't support anyway.
  EXPECT_THAT(first_arg_storage,
              ElementsAre(0x48, 0x13, 0x00, 0x00, 0x89, 0x23, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00));
}

TEST(KernelArgPackingSpecTest, WriteConstantAndAddress) {
  KernelArgPackingSpec first_arg;
  first_arg.WriteArgumentAddress(/*argument_index=*/0);
  first_arg.WriteConstant(static_cast<uint32_t>(0x1234));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<char> first_arg_storage,
                          first_arg.BuildArgument({MakeDevicePointer(0xff42)}));

  EXPECT_THAT(first_arg_storage,
              ElementsAre(0x42, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34,
                          0x12, 0x00, 0x00));
}

TEST(KernelArgPackingSpecTest, WriteConstantWithCustomPacking) {
  KernelArgPackingSpec first_arg;
  first_arg.WriteConstant(CustomData{0x1348});

  TF_ASSERT_OK_AND_ASSIGN(std::vector<char> first_arg_storage,
                          first_arg.BuildArgument(/*args=*/{}));

  // KernelArgPackingSpec::WriteConstant doesn't take endianness into
  // account, so this assertion will fail for big endian architectures - which
  // we don't support anyway.
  EXPECT_THAT(first_arg_storage, ElementsAre(0x49, 0x13, 0x00, 0x00));
}

TEST(KernelArgPackingSpecTest, ToProto) {
  KernelArgPackingSpec first_arg;
  first_arg.WriteConstant(0x1234);
  first_arg.WriteArgumentAddress(/*argument_index=*/0);

  EXPECT_THAT(
      first_arg.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
        relocations { kind: KIND_BITS64_ABSOLUTE argument_index: 0 offset: 4 }
        data: "\x34\x12\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
      )pb")));
}

TEST(KernelArgPackingSpecTest, FromProto) {
  auto proto = ParseTextProtoOrDie<KernelArgPackingSpecProto>(
      R"pb(
        relocations { kind: KIND_BITS64_ABSOLUTE argument_index: 0 offset: 4 }
        data: "\x34\x12\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
      )pb");

  TF_ASSERT_OK_AND_ASSIGN(KernelArgPackingSpec spec,
                          KernelArgPackingSpec::FromProto(proto));
  EXPECT_THAT(spec.BuildArgument({MakeDevicePointer(0xff42)}),
              IsOkAndHolds(ElementsAre(0x34, 0x12, 0x00, 0x00, 0x42, 0xff, 0x00,
                                       0x00, 0x00, 0x00, 0x00, 0x00)));
}

TEST(KernelArgsPackingSpecTest, BuildArguments) {
  KernelArgsPackingSpec spec;
  spec.AddAddressArgument(/*argument_index=*/0);
  spec.AddConstantArgument(0x1234);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<KernelArgsPackedVector> packed_args,
                          spec.BuildArguments({MakeDevicePointer(0xff42)},
                                              /*shared_memory_bytes=*/8989));
  // We expect 3 arguments: 2 parameters and the shared memory which counts as
  // an argument.
  EXPECT_EQ(packed_args->number_of_arguments(), 3);
  EXPECT_EQ(packed_args->number_of_shared_bytes(), 8989);
  EXPECT_EQ(packed_args->argument_addresses().size(), 2);
  EXPECT_THAT(
      absl::Span<const char>(
          absl::bit_cast<const char*>(packed_args->argument_addresses().at(0)),
          8),
      ElementsAre(0x42, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00));
  EXPECT_THAT(
      absl::Span<const char>(
          absl::bit_cast<const char*>(packed_args->argument_addresses().at(1)),
          4),
      ElementsAre(0x34, 0x12, 0x00, 0x00));
}

TEST(KernelArgsPackingSpecTest, ToProto) {
  KernelArgsPackingSpec spec;
  spec.AddAddressArgument(/*argument_index=*/33);
  spec.AddConstantArgument(0x1234);

  EXPECT_THAT(spec.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
                kernel_arguments {
                  relocations {
                    kind: KIND_BITS64_ABSOLUTE
                    argument_index: 33
                    offset: 0
                  }
                  data: "\x00\x00\x00\x00\x00\x00\x00\x00"
                }
                kernel_arguments { data: "\x34\x12\x00\x00" }
              )pb")));
}

TEST(KernelArgsPackingSpecTest, FromProto) {
  auto proto = ParseTextProtoOrDie<KernelArgsPackingSpecProto>(
      R"pb(
        kernel_arguments {
          relocations { kind: KIND_BITS64_ABSOLUTE argument_index: 0 offset: 0 }
          data: "\x00\x00\x00\x00\x00\x00\x00\x00"
        }
        kernel_arguments { data: "\x34\x12\x00\x00" }
      )pb");

  TF_ASSERT_OK_AND_ASSIGN(KernelArgsPackingSpec spec,
                          KernelArgsPackingSpec::FromProto(proto));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<KernelArgsPackedVector> arguments,
                          spec.BuildArguments({MakeDevicePointer(0xff42)},
                                              /*shared_memory_bytes=*/8989));
  // We expect 3 arguments: 2 parameters and the shared memory which counts as
  // an argument.
  EXPECT_EQ(arguments->number_of_arguments(), 3);
  EXPECT_EQ(arguments->number_of_shared_bytes(), 8989);
  ASSERT_THAT(arguments->argument_addresses(), SizeIs(2));
  EXPECT_THAT(absl::Span<const char>(absl::bit_cast<const char*>(
                                         arguments->argument_addresses().at(0)),
                                     8),
              ElementsAre(0x42, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00));
  EXPECT_THAT(absl::Span<const char>(absl::bit_cast<const char*>(
                                         arguments->argument_addresses().at(1)),
                                     4),
              ElementsAre(0x34, 0x12, 0x00, 0x00));
}

}  // namespace
}  // namespace stream_executor
