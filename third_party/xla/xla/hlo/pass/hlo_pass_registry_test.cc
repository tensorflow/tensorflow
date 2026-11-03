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

#include "xla/hlo/pass/hlo_pass_registry.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::NotNull;

struct FakeEnv {
  int device_count = 0;
};

class NoOpPass : public HloModulePass {
 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    return false;
  }
};

class SimplePass : public NoOpPass {
 public:
  absl::string_view name() const override { return "simple-pass"; }
};

class ConfigurablePass : public NoOpPass {
 public:
  ConfigurablePass(int device_count, int64_t threshold)
      : device_count_(device_count), threshold_(threshold) {}

  absl::string_view name() const override { return "configurable-pass"; }
  int device_count() const { return device_count_; }
  int64_t threshold() const { return threshold_; }

 private:
  int device_count_;
  int64_t threshold_;
};

class ConfigurablePassFactory : public HloPassFactory {
 public:
  absl::string_view name() const override { return "configurable-pass"; }

  absl::StatusOr<std::unique_ptr<HloPassInterface>> Create(
      const HloPassContext& context,
      const HloPassOptions& options) const override {
    ABSL_ASSIGN_OR_RETURN(const FakeEnv& env, context.Get<const FakeEnv>());
    return std::make_unique<ConfigurablePass>(
        env.device_count, options.Get<int64_t>("threshold").value_or(10));
  }
};

const HloPassFactory* const kSimplePassRegistration =
    RegisterHloPassFactory<DefaultHloPassFactory<SimplePass>>("simple-pass");
const HloPassFactory* const kConfigurablePassRegistration =
    RegisterHloPassFactory<ConfigurablePassFactory>();

class HloPassRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK(context_.Set(env_));
    ASSERT_OK_AND_ASSIGN(
        configurable_factory_,
        registry_.Register(std::make_unique<ConfigurablePassFactory>()));
  }

  FakeEnv env_{8};
  HloPassContext context_;
  HloPassRegistry registry_;
  const HloPassFactory* configurable_factory_ = nullptr;
};

TEST_F(HloPassRegistryTest, GetReturnsRegisteredFactory) {
  EXPECT_EQ(configurable_factory_->name(), "configurable-pass");
  EXPECT_THAT(registry_.Get("configurable-pass"),
              IsOkAndHolds(configurable_factory_));
}

TEST_F(HloPassRegistryTest, MissingPassReturnsNotFound) {
  EXPECT_THAT(registry_.Get("unknown"),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("'unknown'")));
  EXPECT_THAT(registry_.Create("unknown"),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("'unknown'")));
}

TEST_F(HloPassRegistryTest, DuplicateRegistrationFailsAndKeepsOriginal) {
  EXPECT_THAT(registry_.Register(std::make_unique<ConfigurablePassFactory>()),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("'configurable-pass'")));
  EXPECT_THAT(registry_.Get("configurable-pass"),
              IsOkAndHolds(configurable_factory_));
}

TEST_F(HloPassRegistryTest, CreatesPassWithContextAndOptions) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloPassInterface> pass,
      registry_.Create("configurable-pass", context_, {{"threshold", 64}}));
  const auto* configurable = dynamic_cast<const ConfigurablePass*>(pass.get());
  ASSERT_THAT(configurable, NotNull());
  EXPECT_EQ(configurable->device_count(), 8);
  EXPECT_EQ(configurable->threshold(), 64);
}

TEST_F(HloPassRegistryTest, CreateRejectsUnconsumedOptions) {
  EXPECT_THAT(registry_.Create("configurable-pass", context_,
                               {{"threshold", 64}, {"unknown_option", true}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unknown_option")));
}

TEST_F(HloPassRegistryTest, CreateRejectsMismatchedPassName) {
  ASSERT_OK(registry_.Register(
      std::make_unique<DefaultHloPassFactory<SimplePass>>("misnamed-pass")));

  EXPECT_THAT(registry_.Create("misnamed-pass"),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("simple-pass")));
}

TEST(HloPassRegistryGlobalTest, StaticRegistrationHelpersPopulateRegistry) {
  EXPECT_THAT(HloPassRegistry::Global().Get("simple-pass"),
              IsOkAndHolds(kSimplePassRegistration));
  EXPECT_THAT(HloPassRegistry::Global().Get("configurable-pass"),
              IsOkAndHolds(kConfigurablePassRegistration));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloPassInterface> simple,
                       HloPassRegistry::Global().Create("simple-pass"));
  EXPECT_EQ(simple->name(), "simple-pass");
}

}  // namespace
}  // namespace xla
