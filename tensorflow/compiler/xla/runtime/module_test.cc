/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/runtime/module.h"

#include <memory>
#include <optional>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace runtime {

struct TestStateRef : public Module::StateRef {};

struct TestState : public Module::State {
  explicit TestState(int32_t value) : value(value) {}

  int32_t value;
};

struct ModuleNoStateRef : public StatefulModule<TestState> {
  using Base = StatefulModule<TestState>;

  ModuleNoStateRef() : Base("module-no-state-ref") {}

  absl::StatusOr<std::unique_ptr<TestState>> CreateModuleState() const final {
    return std::make_unique<TestState>(42);
  }
};

struct ModuleWithStateRef : public StatefulModule<TestState, TestStateRef> {
  using Base = StatefulModule<TestState, TestStateRef>;

  ModuleWithStateRef() : Base("module-with-state-ref") {}

  absl::StatusOr<std::unique_ptr<TestState>> CreateModuleState() const final {
    return std::make_unique<TestState>(42);
  }
};

struct ModuleNoState : public StatelessModule {
  ModuleNoState() : StatelessModule("module-no-state") {}
};

TEST(ModuleTest, ModuleNoStateRef) {
  ModuleNoStateRef module;
  EXPECT_EQ(module.name(), "module-no-state-ref");

  auto state = module.CreateModuleState();
  ASSERT_TRUE(state.ok());
  EXPECT_EQ((*state)->value, 42);

  CustomCall::UserData user_data;
  ASSERT_TRUE(module.InitializeUserData(state->get(), user_data).ok());
}

TEST(ModuleTest, ModuleWithStateRef) {
  ModuleWithStateRef module;
  EXPECT_EQ(module.name(), "module-with-state-ref");

  auto state = module.CreateModuleState();
  ASSERT_TRUE(state.ok());
  EXPECT_EQ((*state)->value, 42);

  CustomCall::UserData user_data;
  ASSERT_TRUE(module.InitializeUserData(state->get(), user_data).ok());
}

TEST(ModuleTest, ModuleNoState) {
  ModuleNoState module;
  EXPECT_EQ(module.name(), "module-no-state");

  auto state = dynamic_cast<Module&>(module).CreateState();
  ASSERT_TRUE(state.ok());
  EXPECT_FALSE(state->get());

  CustomCall::UserData user_data;
  ASSERT_TRUE(module.InitializeUserData(user_data).ok());
}

}  // namespace runtime
}  // namespace xla
