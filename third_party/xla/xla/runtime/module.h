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

#ifndef XLA_RUNTIME_MODULE_H_
#define XLA_RUNTIME_MODULE_H_

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/runtime/custom_call_registry.h"

namespace xla {
namespace runtime {

template <typename S, typename R>
class StatefulModule;
class StatelessModule;

//===---------------------------------------------------------------------===///
// Xla runtime module.
//===---------------------------------------------------------------------===///

// Xla runtime module is a way to structure custom calls (external functions)
// together with the state required for calling them. Xla runtime executable can
// be linked with multiple of such modules at run time.
//
// As an example Xla Gpu has a StreamExecutor-based HAL (hardware abstraction
// layer) implemented as a runtime module. Different Xla backends can require
// different sets of modules for running executables, and some of the modules
// can be shared between backends (e.g. XProf integration).
//
// User-defined modules must inherit from `StatefulModule` or `StatelessModule`.
class Module {
 public:
  // When a module is instantiated for each runtime executable it can optionally
  // create a state object that can be used to implement stateful functions,
  // to keep a state between exported functions invocations. State can be
  // accessed from different executable invocations running concurrently.
  struct State {
    virtual ~State() = default;
  };

  // StateRef is a potentially long-lived reference held by the runtime that
  // must guarantee that all pointers added to the user data container are
  // pointing to live objects (see `InitializeUserData` for more details).
  struct StateRef {
    virtual ~StateRef() = default;
  };

  virtual ~Module() = default;

  virtual std::string_view name() const = 0;

  // Creates a new per-executable module state.
  virtual absl::StatusOr<std::unique_ptr<State>> CreateState() const = 0;

  // Initializes `user_data` with values required for calling functions exported
  // from the module. If initializing `user_data` requires creating temporaries,
  // then the returned `StateRef` should guarantee their lifetime. Runtime
  // itself guarantees that `state` is alive when it calls any of the exported
  // functions; this means that `user_data` can safely point to `state`'s
  // members.
  virtual absl::StatusOr<std::unique_ptr<StateRef>> InitializeUserData(
      State* state, CustomCall::UserData& user_data) const = 0;

  // Exports direct custom calls provided by this module to the registry.
  virtual void Export(DirectCustomCallRegistry& registry) const {}

  // Exports dynamic custom calls provided by this module to the registry.
  virtual void Export(DynamicCustomCallRegistry& registry) const {}

 private:
  template <typename S, typename R>
  friend class StatefulModule;
  friend class StatelessModule;

  Module() = default;
};

//===---------------------------------------------------------------------===///
// Stateful and typed Xla runtime module.
//===---------------------------------------------------------------------===///

template <typename S, typename R = void>
class StatefulModule : public Module {
  static_assert(std::is_base_of_v<State, S>,
                "State must be derived from Module::State");

  static_assert(std::is_base_of_v<StateRef, R> || std::is_void_v<R>,
                "State ref must be `void` or derived from Module::StateRef");

 public:
  std::string_view name() const final { return name_; }

  // Creates a new per-executable module State.
  virtual absl::StatusOr<std::unique_ptr<S>> CreateModuleState() const = 0;

  // If state ref is void then UserData initialization must return
  // `absl::Status`, otherwise it must return `absl::StatusOr` with a state
  // reference value.
  virtual std::conditional_t<std::is_void_v<R>, absl::Status,
                             absl::StatusOr<std::unique_ptr<R>>>
  InitializeUserData(S* state, CustomCall::UserData& user_data) const {
    if constexpr (std::is_void_v<R>) {
      return absl::OkStatus();
    } else {
      return nullptr;
    }
  }

 protected:
  explicit StatefulModule(std::string name) : name_(std::move(name)) {}

 private:
  absl::StatusOr<std::unique_ptr<State>> CreateState() const final {
    return CreateModuleState();
  }

  absl::StatusOr<std::unique_ptr<StateRef>> InitializeUserData(
      State* state, CustomCall::UserData& user_data) const final {
    auto initialized = InitializeUserData(dynamic_cast<S*>(state), user_data);

    // Convert `absl::OkStatus` to an `absl::StatusOr` with an empty state ref.
    if constexpr (std::is_void_v<R>) {
      if (initialized.ok()) return nullptr;
    }

    return initialized;
  }

  std::string name_;
};

//===---------------------------------------------------------------------===///
// Stateless Xla runtime module.
//===---------------------------------------------------------------------===///

class StatelessModule : public Module {
 public:
  std::string_view name() const final { return name_; }

  virtual absl::Status InitializeUserData(
      CustomCall::UserData& user_data) const {
    return absl::OkStatus();
  }

 protected:
  explicit StatelessModule(std::string name) : name_(std::move(name)) {}

 private:
  absl::StatusOr<std::unique_ptr<State>> CreateState() const final {
    return nullptr;
  }

  absl::StatusOr<std::unique_ptr<StateRef>> InitializeUserData(
      State* state, CustomCall::UserData& user_data) const final {
    if (auto init = InitializeUserData(user_data); !init.ok()) return init;
    return nullptr;
  }

  std::string name_;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_MODULE_H_
