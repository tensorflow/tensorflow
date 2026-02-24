/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class PjRtTestTopology : public PjRtTopologyDescription {
 public:
  PjRtPlatformId platform_id() const override { return 0; }
  absl::string_view platform_name() const override { return "not_registered"; }
  absl::string_view platform_version() const override { return "test"; }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    LOG(FATAL) << "Unused";
  }
  absl::StatusOr<std::string> Serialize() const override { return "test_topo"; }
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    LOG(FATAL) << "Unused";
  }
  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return absl::UnimplementedError(
        "TestTopology does not support GetDefaultLayout");
  }
};

TEST(PjRtCompilerTest, CompilerNotRegistered) {
  PjRtTestTopology topology;

  CompileOptions options;
  XlaComputation computation;
  auto res = PjRtCompile(options, computation, topology);

  EXPECT_TRUE(absl::IsNotFound(res.status()));
}

TEST(PjRtCompilerTest, CompilerRegistered) {
  class PjRtTestTopology : public PjRtTopologyDescription {
   public:
    PjRtPlatformId platform_id() const override { return 0; }
    absl::string_view platform_name() const override { return "registered"; }
    absl::string_view platform_version() const override { return "test"; }
    std::vector<std::unique_ptr<const PjRtDeviceDescription>>
    DeviceDescriptions() const override {
      LOG(FATAL) << "Unused";
    }
    absl::StatusOr<std::string> Serialize() const override {
      return "test_topo";
    }
    const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
        const override {
      LOG(FATAL) << "Unused";
    }
    absl::StatusOr<Layout> GetDefaultLayout(
        PrimitiveType element_type,
        absl::Span<const int64_t> dims) const override {
      return absl::UnimplementedError(
          "TestTopology does not support GetDefaultLayout");
    }
  };
  PjRtTestTopology topology;

  class PjRtTestCompiler : public PjRtCompiler {
   public:
    absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
        CompileOptions options, const XlaComputation& computation,
        const PjRtTopologyDescription& topology, PjRtClient* client) override {
      return absl::UnimplementedError("test compiler!");
    }
    absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
        CompileOptions options, mlir::ModuleOp module,
        const PjRtTopologyDescription& topology, PjRtClient* client) override {
      return absl::UnimplementedError("test compiler!");
    }
    absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
    DeserializePjRtTopologyDescription(
        const std::string& serialized_topology) override {
      return absl::UnimplementedError("test compiler!");
    }
  };
  CompileOptions options;
  std::unique_ptr<PjRtCompiler> compiler = std::make_unique<PjRtTestCompiler>();
  PjRtRegisterCompiler(topology.platform_name(),
                       options.compiler_variant.value_or(""),
                       std::move(compiler));

  XlaComputation computation;
  auto res = PjRtCompile(options, computation, topology);

  EXPECT_TRUE(absl::IsUnimplemented(res.status()));
}

class PjRtDeserializeTopology : public PjRtTopologyDescription {
 public:
  PjRtPlatformId platform_id() const override { return 0; }
  absl::string_view platform_name() const override {
    return "deserialize_platform";
  }
  absl::string_view platform_version() const override { return "test"; }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    LOG(FATAL) << "Unused";
  }
  absl::StatusOr<std::string> Serialize() const override {
    return "serialized_topology";
  }
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    LOG(FATAL) << "Unused";
  }
  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return absl::UnimplementedError(
        "TestTopology does not support GetDefaultLayout");
  }
};

class PjRtDeserializeCompiler : public PjRtCompiler {
 public:
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override {
    return absl::UnimplementedError("test compiler!");
  }
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override {
    return absl::UnimplementedError("test compiler!");
  }
  absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
  DeserializePjRtTopologyDescription(
      const std::string& serialized_topology) override {
    if (serialized_topology == "serialized_known_topology") {
      return std::make_unique<PjRtDeserializeTopology>();
    }
    return absl::InvalidArgumentError("Unknown topology");
  }
};

class PjRtCompilerDeserializeTopologyTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    auto compiler = std::make_unique<PjRtDeserializeCompiler>();
    PjRtRegisterDefaultCompiler("deserialize_platform", std::move(compiler));
  }
};

TEST_F(PjRtCompilerDeserializeTopologyTest, DeserializeKnownTopology) {
  absl::StatusOr<PjRtCompiler*> compiler_or =
      GetDefaultPjRtCompiler("deserialize_platform");
  EXPECT_OK(compiler_or);
  PjRtCompiler* looked_up_compiler = *compiler_or;

  absl::StatusOr<std::unique_ptr<const PjRtTopologyDescription>> res =
      looked_up_compiler->DeserializePjRtTopologyDescription(
          "serialized_known_topology");
  EXPECT_OK(res);
  EXPECT_EQ((*res)->platform_name(), "deserialize_platform");
}

TEST_F(PjRtCompilerDeserializeTopologyTest, DeserializeUnknownTopology) {
  absl::StatusOr<PjRtCompiler*> compiler_or =
      GetDefaultPjRtCompiler("deserialize_platform");
  EXPECT_OK(compiler_or);
  PjRtCompiler* looked_up_compiler = *compiler_or;

  absl::StatusOr<std::unique_ptr<const PjRtTopologyDescription>> res =
      looked_up_compiler->DeserializePjRtTopologyDescription(
          "unknown_topology");
  EXPECT_TRUE(absl::IsInvalidArgument(res.status()));
}

TEST(PjRtCompilerTest, VariantRegistryLookup) {
  const std::string platform = "variant_test_platform";
  const std::string variant = "test_variant";

  // Register a compiler with a specific variant.
  PjRtRegisterCompiler(platform, variant,
                       std::make_unique<PjRtDeserializeCompiler>());

  // Successful lookup with non-empty variant.
  auto compiler_or = GetPjRtCompiler(platform, variant);
  EXPECT_OK(compiler_or);
  EXPECT_NE(*compiler_or, nullptr);

  // Platform matches but non-empty variant doesn't.
  auto status = GetPjRtCompiler(platform, "wrong_variant");
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound));
  EXPECT_TRUE(absl::IsNotFound(status.status()));

  // Non-empty variant matches but platform doesn't.
  status = GetPjRtCompiler("wrong_platform", variant);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound));
  EXPECT_TRUE(absl::IsNotFound(status.status()));

  // Lookup using the single-parameter overload.
  status = GetDefaultPjRtCompiler(platform);
  EXPECT_TRUE(absl::IsNotFound(status.status()));
}

TEST(PjRtCompilerTest, CompilerFactoryRegistered) {
  const std::string platform = "factory_test_platform";
  const std::string variant = "factory_variant";
  auto factory_called = std::make_shared<bool>(false);

  auto factory = [&]() -> absl::StatusOr<std::unique_ptr<PjRtCompiler>> {
    *factory_called = true;
    return std::make_unique<PjRtDeserializeCompiler>();
  };

  PjRtRegisterCompilerFactory(platform, variant, std::move(factory));

  class PjRtResetPlatformNameTopology : public PjRtTestTopology {
   public:
    absl::string_view platform_name() const override {
      return "factory_test_platform";
    }
  };
  PjRtResetPlatformNameTopology topology;
  CompileOptions options;
  options.compiler_variant = variant;
  XlaComputation computation;

  // Factory should not be called yet.
  EXPECT_FALSE(*factory_called);

  // PjRtCompile should trigger the factory via GetOrCreateCompiler.
  auto res = PjRtCompile(options, computation, topology);

  EXPECT_TRUE(*factory_called);
  // PjRtDeserializeCompiler::Compile returns Unimplemented("test compiler!").
  EXPECT_TRUE(absl::IsUnimplemented(res.status()));
}

TEST(PjRtCompilerInitializationTest, InitializeCompilerVariant) {
  const std::string platform = "init_test_platform";
  const std::string variant = "init_variant";
  auto factory_call_count = std::make_shared<int>(0);

  PjRtCompilerRegistry compiler_registry;

  EXPECT_OK(compiler_registry.RegisterFactory(
      platform, variant,
      [&]() -> absl::StatusOr<std::unique_ptr<PjRtCompiler>> {
        (*factory_call_count)++;
        return std::make_unique<PjRtDeserializeCompiler>();
      }));

  EXPECT_EQ(*factory_call_count, 0);

  // First call should invoke the factory.
  auto status = compiler_registry.InitializeVariant(platform, variant);
  EXPECT_OK(status);
  EXPECT_EQ(*factory_call_count, 1);

  // Second call should be a no-op as the compiler is already in the registry.
  status = compiler_registry.InitializeVariant(platform, variant);
  EXPECT_OK(status);
  EXPECT_EQ(*factory_call_count, 1);

  // Verify it's actually registered by looking it up.
  auto compiler = compiler_registry.GetCompiler(platform, variant);
  EXPECT_OK(compiler);
  EXPECT_NE(*compiler, nullptr);
}

TEST(PjRtCompilerInitializationTest, InitializeCompilerVariants) {
  const std::string platform_1 = "platform_1";
  const std::string platform_2 = "platform_2";
  const std::string variant_1 = "variant_1";
  const std::string variant_2 = "variant_2";
  auto factory_call_count_1 = std::make_shared<int>(0);
  auto factory_call_count_2 = std::make_shared<int>(0);

  PjRtCompilerRegistry compiler_registry;

  // Register two different compiler factories.
  EXPECT_OK(compiler_registry.RegisterFactory(
      platform_1, variant_1,
      [factory_call_count_1]()
          -> absl::StatusOr<std::unique_ptr<PjRtCompiler>> {
        (*factory_call_count_1)++;
        return std::make_unique<PjRtDeserializeCompiler>();
      }));

  EXPECT_OK(compiler_registry.RegisterFactory(
      platform_2, variant_2,
      [factory_call_count_2]()
          -> absl::StatusOr<std::unique_ptr<PjRtCompiler>> {
        (*factory_call_count_2)++;
        return std::make_unique<PjRtDeserializeCompiler>();
      }));

  // Verify factories haven't been called yet.
  EXPECT_EQ(*factory_call_count_1, 0);
  EXPECT_EQ(*factory_call_count_2, 0);

  // Initialize all registered variants.
  auto status = compiler_registry.InitializeAllVariants();
  EXPECT_OK(status);

  // Both new factories should have been invoked exactly once.
  EXPECT_EQ(*factory_call_count_1, 1);
  EXPECT_EQ(*factory_call_count_2, 1);

  // Verify the compilers are now available in the registry.
  auto compiler1 = compiler_registry.GetCompiler(platform_1, variant_1);
  EXPECT_OK(compiler1);
  EXPECT_NE(*compiler1, nullptr);

  auto compiler2 = compiler_registry.GetCompiler(platform_2, variant_2);
  EXPECT_OK(compiler2);
  EXPECT_NE(*compiler2, nullptr);
}

TEST(PjRtCompilerInitializationTest, FactoryError) {
  const std::string platform = "error_test_platform";
  const std::string variant = "error_variant";

  PjRtCompilerRegistry compiler_registry;

  EXPECT_OK(compiler_registry.RegisterFactory(
      platform, variant, []() -> absl::StatusOr<std::unique_ptr<PjRtCompiler>> {
        return absl::InternalError("factory failed");
      }));

  // The error from the factory should be propagated.
  auto status = compiler_registry.InitializeVariant(platform, variant);
  EXPECT_TRUE(absl::IsInternal(status));
  EXPECT_EQ(status.message(), "factory failed");
}

TEST(PjRtCompilerInitializationTest, UnknownVariantError) {
  const std::string platform = "unknown_test_platform";
  const std::string variant = "no_such_variant";

  PjRtCompilerRegistry compiler_registry;

  // Requesting a variant that has no factory and no registered compiler.
  auto status = compiler_registry.InitializeVariant(platform, variant);
  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr("No compiler factory for platform: "
                         "unknown_test_platform, variant: no_such_variant")));
}

TEST(PjRtCompilerInitializationTest, RegistriesOutOfSync) {
  const std::string platform = "sync_test_platform";
  const std::string variant = "";  // Default variant
  auto factory_called = std::make_shared<bool>(false);
  PjRtCompilerRegistry compiler_registry;

  // Manually register a compiler in the CompilerRegistry.
  EXPECT_OK(compiler_registry.RegisterCompiler(
      platform, variant, std::make_unique<PjRtDeserializeCompiler>()));

  // Register a factory for the same key in the CompilerFactoryRegistry.
  // This simulates the "out of sync" state where both maps have entries,
  // but the compiler is already "initialized".
  EXPECT_OK(compiler_registry.RegisterFactory(
      platform, variant,
      [&]() -> absl::StatusOr<std::unique_ptr<PjRtCompiler>> {
        *factory_called = true;
        return std::make_unique<PjRtDeserializeCompiler>();
      }));

  // Initialize/Access the compiler.
  auto status = compiler_registry.InitializeVariant(platform, variant);
  EXPECT_OK(status);

  // Verify that the existing compiler was used and the factory was NOT called.
  // GetOrCreateCompiler should find the entry in CompilerRegistry() first.
  EXPECT_FALSE(*factory_called);
}

}  // namespace
}  // namespace xla
