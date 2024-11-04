/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_ACCELERATION_TEST_UTIL_INTERNAL_H_
#define TENSORFLOW_LITE_KERNELS_ACCELERATION_TEST_UTIL_INTERNAL_H_

#include <algorithm>
#include <atomic>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "re2/re2.h"

namespace tflite {

// Reads the acceleration configuration, handles comments and empty lines and
// the basic data conversion format (split into key, value, recognition of
// the line being a white or black list entry) and gives the data to the
// consumer to be inserted into the target collection.
void ReadAccelerationConfig(
    const char* config,
    const std::function<void(std::string, std::string, bool)>& consumer);

template <typename T>
class ConfigurationEntry {
 public:
  ConfigurationEntry(const std::string& test_id_rex, T test_config,
                     bool is_denylist)
      : test_id_rex_(new RE2(test_id_rex)),
        test_config_(test_config),
        is_denylist_(is_denylist) {}

  bool Matches(const std::string& test_id) const {
    return RE2::FullMatch(test_id, *test_id_rex_);
  }
  bool IsDenylistEntry() const { return is_denylist_; }
  const T& TestConfig() const { return test_config_; }

  const std::string& TestIdRex() const { return test_id_rex_->pattern(); }

 private:
  std::unique_ptr<RE2> test_id_rex_;
  T test_config_;
  bool is_denylist_;
};

// Returns the acceleration test configuration for the given test id and
// the given acceleration configuration type.
// The configuration type is responsible of providing the test configuration
// and the parse function to convert configuration lines into configuration
// objects.
template <typename T>
std::optional<T> GetAccelerationTestParam(std::string test_id) {
  static std::atomic<std::vector<ConfigurationEntry<T>>*> test_config_ptr;

  if (test_config_ptr.load() == nullptr) {
    auto config = new std::vector<ConfigurationEntry<T>>();

    auto consumer = [&config](std::string key, std::string value_str,
                              bool is_denylist) mutable {
      T value = T::ParseConfigurationLine(value_str);
      config->emplace_back(key, value, is_denylist);
    };

    ReadAccelerationConfig(T::AccelerationTestConfig(), consumer);

    // Even if it has been already set, it would be just replaced with the
    // same value, just freeing the old value to avoid leaks
    auto* prev_val = test_config_ptr.exchange(config);
    delete prev_val;
  }

  const std::vector<ConfigurationEntry<T>>* test_config =
      test_config_ptr.load();

  const auto test_config_iter =
      std::find_if(test_config->begin(), test_config->end(),
                   [&test_id](const ConfigurationEntry<T>& elem) {
                     return elem.Matches(test_id);
                   });
  if (test_config_iter != test_config->end() &&
      !test_config_iter->IsDenylistEntry()) {
    return std::optional<T>(test_config_iter->TestConfig());
  } else {
    return std::optional<T>();
  }
}

}  //  namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_ACCELERATION_TEST_UTIL_INTERNAL_H_
