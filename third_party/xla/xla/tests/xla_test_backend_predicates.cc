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
#include "xla/tests/xla_test_backend_predicates.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace xla::test {
namespace {

absl::string_view GetXlaTestDevice() { return GetEnvOrDie("XLA_TEST_DEVICE"); }
absl::string_view GetXlaTestDeviceType() {
  return GetEnvOrDie("XLA_TEST_DEVICE_TYPE");
}

std::vector<absl::string_view> GetXlaTestModifiers() {
  return absl::StrSplit(GetEnvOrDie("XLA_TEST_MODIFIERS"), ',');
}
}  // namespace

absl::string_view GetEnvOrDie(const char* key) {
  const char* val = std::getenv(key);
  CHECK_NE(val, nullptr)
      << key
      << " not set! Make sure your test target is defined by a macro which "
         "sets the environment appropriately (e.g. `xla_test`).";
  return val;
}

bool DeviceIs(absl::string_view device) { return device == GetXlaTestDevice(); }

bool DeviceIsOneOf(absl::Span<const absl::string_view> devices) {
  for (const absl::string_view device : devices) {
    if (DeviceIs(device)) {
      return true;
    }
  }
  return false;
}

bool DeviceTypeIs(absl::string_view device) {
  return device == GetXlaTestDeviceType();
}

bool DeviceTypeIsOneOf(absl::Span<const absl::string_view> devices) {
  for (const absl::string_view device : devices) {
    if (DeviceTypeIs(device)) {
      return true;
    }
  }
  return false;
}

bool HasModifiers(absl::Span<const absl::string_view> modifiers) {
  std::vector<absl::string_view> set_modifiers = GetXlaTestModifiers();
  for (const absl::string_view m : modifiers) {
    if (std::find(set_modifiers.begin(), set_modifiers.end(), m) ==
        set_modifiers.end()) {
      return false;
    }
  }
  return true;
}

bool BackendLike(absl::string_view device,
                 absl::Span<const absl::string_view> modifiers) {
  return DeviceIs(device) && HasModifiers(modifiers);
}

bool BackendIsExactly(absl::string_view device,
                      absl::Span<const absl::string_view> modifiers) {
  bool device_matches = DeviceIs(device);

  std::vector<absl::string_view> set_modifiers = GetXlaTestModifiers();
  bool modifiers_match = absl::flat_hash_set<absl::string_view>(
                             set_modifiers.begin(), set_modifiers.end()) ==
                         absl::flat_hash_set<absl::string_view>(
                             modifiers.begin(), modifiers.end());

  return device_matches && modifiers_match;
}

// Returns true only for base variant hardware + emulation.
bool BackendIsStrict(absl::string_view device) {
  const bool device_matches = DeviceIs(device);

  std::vector<absl::string_view> modifiers = GetXlaTestModifiers();
  const bool modifiers_match =
      modifiers.size() == 1 && (modifiers[0] == kHardware ||
                                modifiers[0] == kIss || modifiers[0] == kGrm);
  return device_matches && modifiers_match;
}

}  // namespace xla::test
