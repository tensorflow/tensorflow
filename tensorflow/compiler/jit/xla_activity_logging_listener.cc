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

#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"

namespace tensorflow {
namespace {

// Listens to XLA activity and logs them using tensorflow::Logger.
class XlaActivityLoggingListener final : public XlaActivityListener {
 public:
  absl::Status Listen(
      const XlaAutoClusteringActivity& auto_clustering_activity) override {
    if (!IsEnabled()) {
      VLOG(3) << "Logging XlaAutoClusteringActivity disabled";
      return absl::OkStatus();
    }

    return absl::OkStatus();
  }

  absl::Status Listen(
      const XlaJitCompilationActivity& jit_compilation_activity) override {
    if (!IsEnabled()) {
      VLOG(3) << "Logging XlaJitCompilationActivity disabled";
      return absl::OkStatus();
    }

    return absl::OkStatus();
  }

  absl::Status Listen(
      const XlaOptimizationRemark& optimization_remark) override {
    if (!IsEnabled()) {
      VLOG(3) << "Logging XlaJitCompilationActivity disabled";
      return absl::OkStatus();
    }

    return absl::OkStatus();
  }

 private:
  bool IsEnabled() {
    static bool result = ComputeIsEnabled();
    return result;
  }

  bool ComputeIsEnabled() {
    char* log_xla_activity = getenv("TF_LOG_XLA_ACTIVITY");
    if (log_xla_activity == nullptr) {
      bool enabled_by_default = true;
      return enabled_by_default;
    }

    return absl::string_view(log_xla_activity) == "1";
  }
};

bool Register() {
  RegisterXlaActivityListener(std::make_unique<XlaActivityLoggingListener>());
  return false;
}

bool unused = Register();
}  // namespace
}  // namespace tensorflow
