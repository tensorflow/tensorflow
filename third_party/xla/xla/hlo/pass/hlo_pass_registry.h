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

#ifndef XLA_HLO_PASS_HLO_PASS_REGISTRY_H_
#define XLA_HLO_PASS_HLO_PASS_REGISTRY_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"

namespace xla {

// Factory for instantiating an HloPassInterface from environment objects.
class HloPassFactory {
 public:
  virtual ~HloPassFactory() = default;

  // Pass name used in pipeline configuration.
  virtual absl::string_view name() const = 0;

  virtual absl::StatusOr<std::unique_ptr<HloPassInterface>> Create(
      const HloPassContext& context, const HloPassOptions& options) const = 0;
};

// Thread-safe registry of HloPassFactory instances keyed by pass name.
class HloPassRegistry {
 public:
  static HloPassRegistry& Global();

  // Registers `factory` under `factory->name()`. Returns AlreadyExistsError if
  // a factory with that name is already registered.
  absl::StatusOr<const HloPassFactory* absl_nonnull> Register(
      std::unique_ptr<HloPassFactory> factory);

  absl::StatusOr<const HloPassFactory* absl_nonnull> Get(
      absl::string_view name) const;

  // Instantiates the pass registered under `name`, verifies that all `options`
  // were consumed, and checks that `pass->name() == name`.
  absl::StatusOr<std::unique_ptr<HloPassInterface>> Create(
      absl::string_view name, const HloPassContext& context = {},
      const HloPassOptions& options = {}) const;

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<HloPassFactory>> factories_
      ABSL_GUARDED_BY(mu_);
};

// Default HloPassFactory for passes that take no arguments.
template <typename PassT>
class DefaultHloPassFactory final : public HloPassFactory {
 public:
  explicit DefaultHloPassFactory(absl::string_view name) : name_(name) {}

  absl::string_view name() const override { return name_; }

  absl::StatusOr<std::unique_ptr<HloPassInterface>> Create(
      const HloPassContext&, const HloPassOptions&) const override {
    return std::make_unique<PassT>();
  }

 private:
  std::string name_;
};

// Registers a pass factory. Crashes if the name is already registered.
template <typename FactoryT, typename... Args>
const HloPassFactory* absl_nonnull RegisterHloPassFactory(Args&&... args) {
  absl::StatusOr<const HloPassFactory* absl_nonnull> factory =
      HloPassRegistry::Global().Register(
          std::make_unique<FactoryT>(std::forward<Args>(args)...));
  CHECK_OK(factory);
  return *factory;
}

}  // namespace xla

#endif  // XLA_HLO_PASS_HLO_PASS_REGISTRY_H_
