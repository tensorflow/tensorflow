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

#include "xla/backends/cpu/runtime/msan_emulated_tls.h"

#include <cstdint>
#include <thread>  // NOLINT(build/c++11)

#include <gtest/gtest.h>
#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/container/flat_hash_set.h"

namespace xla::cpu {
namespace {

#ifdef ABSL_HAVE_MEMORY_SANITIZER
constexpr bool kHasMemorySanitizer = true;
#else
constexpr bool kHasMemorySanitizer = false;
#endif

TEST(MsanEmulatedTlsTest, AllSelectorsReturnDistinctAddresses) {
  if constexpr (!kHasMemorySanitizer) {
    GTEST_SKIP() << "This test requires an msan build";
  } else {
    absl::flat_hash_set<void*> addresses;
    for (uintptr_t selector =
             static_cast<uintptr_t>(MsanTlsSelector::kParamTls);
         selector <= static_cast<uintptr_t>(MsanTlsSelector::kOriginTls);
         ++selector) {
      void* address = __xla_cpu_runtime_emutls_get_address(
          reinterpret_cast<void*>(selector));
      EXPECT_NE(address, nullptr) << "selector " << selector;
      EXPECT_TRUE(addresses.insert(address).second) << "selector " << selector;
    }
  }
}

TEST(MsanEmulatedTlsTest, AddressesAreThreadLocal) {
  if constexpr (!kHasMemorySanitizer) {
    GTEST_SKIP() << "This test requires an msan build";
  } else {
    void* selector = reinterpret_cast<void*>(
        static_cast<uintptr_t>(MsanTlsSelector::kParamTls));
    void* address = __xla_cpu_runtime_emutls_get_address(selector);
    void* other_thread_address = nullptr;
    std::thread thread([&] {
      other_thread_address = __xla_cpu_runtime_emutls_get_address(selector);
    });
    thread.join();
    EXPECT_NE(other_thread_address, nullptr);
    EXPECT_NE(other_thread_address, address);
  }
}

TEST(MsanEmulatedTlsDeathTest, InvalidSelectorDies) {
  for (uintptr_t selector : {uintptr_t{0}, uintptr_t{9}}) {
    EXPECT_DEATH(
        __xla_cpu_runtime_emutls_get_address(reinterpret_cast<void*>(selector)),
        "Invalid msan TLS selector");
  }
}

}  // namespace
}  // namespace xla::cpu
