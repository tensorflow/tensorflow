/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt_proxy/common/test_utils.h"

#include <functional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/debugging/leak_check.h"
#include "absl/synchronization/mutex.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

class Overrides {
 public:
  void Set(TestHookName h, std::function<void(bool*)> fn) {
    absl::MutexLock l(&mu_);
    overrides_[h] = std::move(fn);
  }

  void Clear(TestHookName h) {
    absl::MutexLock l(&mu_);
    overrides_.erase(h);
  }

  void Call(TestHookName h, bool* param1) {
    absl::MutexLock l(&mu_);
    const auto it = overrides_.find(h);
    if (it != overrides_.end()) {
      it->second(param1);
    }
  }

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<TestHookName, std::function<void(bool*)>> overrides_
      ABSL_GUARDED_BY(mu_);
};

Overrides* overrides() {
  // Declaring a global absl::NoDestructor<Overrides> is easier, but as of Sep
  // 2024, NoDestructor<> was not yet available in the version of absl linked
  // into TSL.
  static Overrides* result = []() {
    auto* result = new Overrides;
    absl::IgnoreLeak(result);
    return result;
  }();
  return result;
}

};  // namespace

void TestHookSet(TestHookName h, std::function<void(bool*)> fn) {
  overrides()->Set(h, std::move(fn));
}
void TestHookClear(TestHookName h) { overrides()->Clear(h); }

void TestHookCall(TestHookName h, bool* param1) {
  overrides()->Call(h, param1);
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
