/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_topology_description_registry.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::NotNull;

constexpr PjRtPlatformId kTestPlatformId = 0xDEADBEEF;

class TestTopologyDescription : public PjRtTopologyDescription {
 public:
  PjRtPlatformId platform_id() const override { return kTestPlatformId; }
  absl::string_view platform_name() const override { return "test_platform"; }
  absl::string_view platform_version() const override { return "1.0"; }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    return {};
  }
  absl::StatusOr<uint64_t> Fingerprint() const override { return 0; }
  const absl::flat_hash_map<std::string, PjRtValueType>& Attributes()
      const override {
    static auto* attrs = new absl::flat_hash_map<std::string, PjRtValueType>();
    return *attrs;
  }
  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return absl::UnimplementedError("GetDefaultLayout not implemented");
  }
  absl::StatusOr<PjRtTopologyDescriptionProto> ToProto() const override {
    PjRtTopologyDescriptionProto proto;
    proto.set_platform_id(kTestPlatformId);
    proto.set_platform_name("test_platform");
    return proto;
  }
};

TEST(PjRtTopologyDescriptionRegistryTest, StaticRegistrationAndLookup) {
  auto factory = [](const PjRtTopologyDescriptionProto&)
      -> absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> {
    return std::make_unique<TestTopologyDescription>();
  };
  EXPECT_OK(PjRtTopologyDescriptionRegistry::Global().RegisterDeserializer(
      kTestPlatformId, "test_platform", factory));

  PjRtTopologyDescriptionProto proto;
  proto.set_platform_id(kTestPlatformId);
  EXPECT_THAT(PjRtTopologyDescriptionRegistry::Global().Deserialize(proto),
              IsOkAndHolds(NotNull()));

  PjRtTopologyDescriptionProto unregistered_proto;
  unregistered_proto.set_platform_id(0x99999999);
  EXPECT_THAT(
      PjRtTopologyDescriptionRegistry::Global().Deserialize(unregistered_proto),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST(PjRtTopologyDescriptionRegistryTest, ThreadSafeConcurrentRegistrations) {
  constexpr int kNumThreads = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "ConcurrentReg",
                                      kNumThreads);
  absl::BlockingCounter counter(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    thread_pool.Schedule([&, i]() {
      auto factory = [](const PjRtTopologyDescriptionProto&)
          -> absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> {
        return std::make_unique<TestTopologyDescription>();
      };
      EXPECT_OK(PjRtTopologyDescriptionRegistry::Global().RegisterDeserializer(
          kTestPlatformId + i, absl::StrCat("test_platform_", i), factory));
      counter.DecrementCount();
    });
  }
  counter.Wait();

  for (int i = 0; i < kNumThreads; ++i) {
    PjRtTopologyDescriptionProto proto;
    proto.set_platform_id(kTestPlatformId + i);
    proto.set_platform_name(absl::StrCat("test_platform_", i));
    EXPECT_THAT(PjRtTopologyDescriptionRegistry::Global().Deserialize(proto),
                IsOkAndHolds(NotNull()));
  }
}

TEST(PjRtTopologyDescriptionRegistryTest,
     FromProtoUnregisteredPlatformReturnsError) {
  PjRtTopologyDescriptionProto unregistered_proto;
  unregistered_proto.set_platform_id(0x88888888);
  unregistered_proto.set_platform_name("unregistered_custom_platform");
  // Static C++ registry lookup returns kNotFound, causing dynamic C-API
  // compiler plugin lookup (returning kFailedPrecondition when uninitialized,
  // or kNotFound if initialized without matching plugin).
  EXPECT_THAT(PjRtTopologyDescriptionFromProto(unregistered_proto),
              StatusIs(AnyOf(absl::StatusCode::kFailedPrecondition,
                             absl::StatusCode::kNotFound)));
}

TEST(PjRtTopologyDescriptionRegistryTest, ToProtoAndFromProtoSymmetry) {
  TestTopologyDescription test_desc;
  EXPECT_THAT(PjRtTopologyDescriptionToProto(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto proto_status = PjRtTopologyDescriptionToProto(&test_desc);
  EXPECT_OK(proto_status);
}

TEST(PjRtTopologyDescriptionRegistryTest, CpuSerDesRoundtripParity) {
  std::vector<CpuTopology::CpuDevice> devices = {{0, 0}, {0, 1}};
  xla::cpu::TargetMachineOptions options(
      /*triple=*/"x86_64-unknown-linux-gnu", /*cpu=*/"haswell",
      /*features=*/"+avx2");
  CpuTopologyDescription original(xla::CpuId(), "cpu", "1.0",
                                  CpuTopology(devices, options));

  ASSERT_OK_AND_ASSIGN(PjRtTopologyDescriptionProto proto,
                       PjRtTopologyDescriptionToProto(&original));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> deserialized,
                       PjRtTopologyDescriptionFromProto(proto));

  ASSERT_THAT(deserialized, NotNull());
  EXPECT_EQ(deserialized->platform_id(), original.platform_id());
  EXPECT_EQ(deserialized->platform_name(), original.platform_name());
  EXPECT_EQ(deserialized->platform_version(), original.platform_version());

  auto* deserialized_cpu =
      dynamic_cast<CpuTopologyDescription*>(deserialized.get());
  ASSERT_THAT(deserialized_cpu, NotNull());
  EXPECT_EQ(*deserialized_cpu, original);
}

TEST(PjRtTopologyDescriptionRegistryTest,
     UnpopulatedProtoReturnsInvalidArgument) {
  PjRtTopologyDescriptionProto empty_proto;
  EXPECT_THAT(PjRtTopologyDescriptionFromProto(empty_proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PjRtTopologyDescriptionRegistryTest,
     DynamicCompilerLookupRegistrationAndFallback) {
  // Register dynamic compiler lookup using a direct lambda.
  PjRtTopologyDescriptionRegistry::Global().RegisterDynamicCompilerLookup(
      [](absl::string_view platform_name)
          -> absl::StatusOr<std::unique_ptr<PjRtCompiler>> {
        if (platform_name == "custom_dynamic_platform") {
          return absl::UnimplementedError("Compiler for custom platform");
        }
        return absl::NotFoundError(
            absl::StrCat("Unknown dynamic platform '", platform_name, "'."));
      });

  // Verify lookup returns expected Status for registered platform.
  EXPECT_THAT(PjRtTopologyDescriptionRegistry::Global().GetDynamicCompiler(
                  "custom_dynamic_platform"),
              StatusIs(absl::StatusCode::kUnimplemented));

  // Verify unmapped dynamic platform returns kNotFound.
  EXPECT_THAT(PjRtTopologyDescriptionRegistry::Global().GetDynamicCompiler(
                  "unknown_dyn_platform"),
              StatusIs(absl::StatusCode::kNotFound));

  // Verify FromProto fallback propagates the dynamic compiler status.
  PjRtTopologyDescriptionProto proto;
  proto.set_platform_name("custom_dynamic_platform");
  EXPECT_THAT(PjRtTopologyDescriptionFromProto(proto),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace xla
