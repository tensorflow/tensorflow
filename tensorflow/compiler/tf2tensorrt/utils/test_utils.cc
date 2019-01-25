/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/utils/test_utils.h"

#include <unordered_map>
#include <vector>

#include "re2/re2.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace tensorrt {
namespace test {

// TODO(aaroey): make this class thread-safe.
class TestValueManager {
 public:
  static TestValueManager* singleton() {
    static TestValueManager* manager = new TestValueManager();
    return manager;
  }

  void Enable() {
    VLOG(1) << "Enabling test value";
    enabled_ = true;
  }

  void Add(const string& label, const string& value) {
    if (TF_PREDICT_FALSE(enabled_)) {
      QCHECK_NE("", value);
      VLOG(1) << "Adding test value: " << label << " -> " << value;
      values_.insert({label, value});
    }
  }

  string Get(const string& label) {
    if (TF_PREDICT_FALSE(enabled_)) {
      VLOG(1) << "Getting test value by " << label;
      auto itr = values_.find(label);
      if (itr == values_.end()) return "";
      return itr->second;
    }
    return "";
  }

  void Clear(const string& pattern) {
    if (TF_PREDICT_FALSE(enabled_)) {
      VLOG(1) << "Clearing test values";
      if (pattern.empty()) {
        values_.clear();
        return;
      }
      std::vector<string> keys_to_clear;
      for (const auto& kv : values_) {
        if (RE2::FullMatch(kv.first, pattern)) {
          keys_to_clear.push_back(kv.first);
        }
      }
      for (const string& key : keys_to_clear) {
        values_.erase(key);
      }
    }
  }

 private:
  TestValueManager() : enabled_(false) {}

  bool enabled_;
  std::unordered_map<string, string> values_;
};

void EnableTestValue() { TestValueManager::singleton()->Enable(); }

void ClearTestValues(const string& pattern) {
  TestValueManager::singleton()->Clear(pattern);
}

void AddTestValue(const string& label, const string& value) {
  TestValueManager::singleton()->Add(label, value);
}

string GetTestValue(const string& label) {
  return TestValueManager::singleton()->Get(label);
}

}  // namespace test
}  // namespace tensorrt
}  // namespace tensorflow
