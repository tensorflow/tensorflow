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

#include "xla/stream_executor/tpu/tpu_initialize_util.h"

#include <unistd.h>

#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/test.h"

namespace tensorflow {
namespace tpu {
namespace {

// RAII helper to scope global environment variable changes in unit tests.
class EnvVarGuard {
 public:
  EnvVarGuard(absl::string_view name, absl::string_view val) : name_(name) {
    const char* original = getenv(name_.c_str());
    if (original != nullptr) {
      original_val_ = original;
      has_original_ = true;
    }
    setenv(name_.c_str(), std::string(val).c_str(), 1);
  }
  EnvVarGuard(const EnvVarGuard&) = delete;
  EnvVarGuard& operator=(const EnvVarGuard&) = delete;
  EnvVarGuard(EnvVarGuard&&) = delete;
  EnvVarGuard& operator=(EnvVarGuard&&) = delete;

  ~EnvVarGuard() {
    if (has_original_) {
      setenv(name_.c_str(), original_val_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string original_val_;
  bool has_original_ = false;
};

// RAII helper to scope unsetting environment variables in unit tests.
class EnvVarUnsetGuard {
 public:
  explicit EnvVarUnsetGuard(absl::string_view name) : name_(name) {
    const char* original = getenv(name_.c_str());
    if (original != nullptr) {
      original_val_ = original;
      has_original_ = true;
    }
    unsetenv(name_.c_str());
  }
  EnvVarUnsetGuard(const EnvVarUnsetGuard&) = delete;
  EnvVarUnsetGuard& operator=(const EnvVarUnsetGuard&) = delete;
  EnvVarUnsetGuard(EnvVarUnsetGuard&&) = delete;
  EnvVarUnsetGuard& operator=(EnvVarUnsetGuard&&) = delete;

  ~EnvVarUnsetGuard() {
    if (has_original_) {
      setenv(name_.c_str(), original_val_.c_str(), 1);
    }
  }

 private:
  std::string name_;
  std::string original_val_;
  bool has_original_ = false;
};

TEST(TpuInitializeUtilTest, TryAcquireTpuLockEnvOverrideForceLoad) {
  ResetTpuLockStateForTesting();
  EnvVarGuard loader_guard("TPU_LOAD_LIBRARY", "1");
  absl::Status status = TryAcquireTpuLock();
  EXPECT_TRUE(status.ok());
}

TEST(TpuInitializeUtilTest, TryAcquireTpuLockEnvOverrideNoLoad) {
  ResetTpuLockStateForTesting();
  EnvVarGuard loader_guard("TPU_LOAD_LIBRARY", "0");
  absl::Status status = TryAcquireTpuLock();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

TEST(TpuInitializeUtilTest, TryAcquireTpuLockAllowMultiple) {
  ResetTpuLockStateForTesting();
  EnvVarUnsetGuard loader_guard("TPU_LOAD_LIBRARY");
  EnvVarGuard multiple_guard("ALLOW_MULTIPLE_LIBTPU_LOAD", "true");
  absl::Status status = TryAcquireTpuLock();
  EXPECT_TRUE(status.ok());
}

TEST(TpuInitializeUtilTest, TryAcquireTpuLockStandardFlow) {
  ResetTpuLockStateForTesting();
  EnvVarUnsetGuard loader_guard("TPU_LOAD_LIBRARY");
  EnvVarUnsetGuard multiple_guard("ALLOW_MULTIPLE_LIBTPU_LOAD");
  EnvVarGuard bounds_guard("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1");

  // Should acquire the lock successfully during the first attempt.
  absl::Status status = TryAcquireTpuLock();
  EXPECT_TRUE(status.ok());

  // Subsequent calls should return ok due to the cached fast-path check.
  absl::Status status_cached = TryAcquireTpuLock();
  EXPECT_TRUE(status_cached.ok());
}

}  // namespace
}  // namespace tpu
}  // namespace tensorflow
