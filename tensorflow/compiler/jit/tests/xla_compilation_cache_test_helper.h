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
#ifndef TENSORFLOW_COMPILER_JIT_TESTS_XLA_COMPILATION_CACHE_TEST_HELPER_H_
#define TENSORFLOW_COMPILER_JIT_TESTS_XLA_COMPILATION_CACHE_TEST_HELPER_H_

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// A listener to inspect the use of XLA's persistent compilation cache entries.
class JitCompilationListener : public XlaActivityListener {
 public:
  Status Listen(
      const XlaAutoClusteringActivity& auto_clustering_activity) override {
    return OkStatus();
  }

  Status Listen(
      const XlaJitCompilationActivity& jit_compilation_activity) override {
    used_persistent_cache_.push_back(
        jit_compilation_activity.used_persistent_cache());
    return OkStatus();
  }

  Status Listen(const XlaOptimizationRemark& optimization_remark) override {
    return OkStatus();
  }

  ~JitCompilationListener() override {}

  Status VerifyListenerHistory(bool expect_persistent_cache_use) {
    for (bool used_persistent_cache : used_persistent_cache_) {
      if (used_persistent_cache != expect_persistent_cache_use) {
        return errors::FailedPrecondition("Unexpected listener history.");
      }
    }
    return OkStatus();
  }

  void ClearListenerHistory() { used_persistent_cache_.clear(); }

 private:
  std::vector<bool> used_persistent_cache_;
};

// Fixture for testing XLA compilation cache serialization.
class XlaCompilationCacheSerializeTest : public ::testing::Test {
 protected:
  XlaCompilationCacheSerializeTest() {
    auto listener = absl::make_unique<JitCompilationListener>();
    listener_ = listener.get();
    RegisterXlaActivityListener(std::move(listener));
  }

  JitCompilationListener* listener() const { return listener_; }

  // Returns a test graph that will split into two XLA clusters (due to a node
  // with _XlaCompile = false).
  GraphDef GetTestGraph(const PartialTensorShape& input_shape);

  // Runs the graph using specified batch size both with and without XLA JIT
  // compilation. Returns an error if the results between the two do not match.
  Status ExecuteWithBatch(const GraphDef& graph, int batch);

  // Adds the suffix "_altered" to the HLO module names of all of the persistent
  // XLA compilation cache entries found at the specified directory. If none are
  // found, returns NOT_FOUND error.
  Status AlterPersistentCacheEntryHloModuleNames(
      absl::string_view persistent_cache_dir_path,
      absl::string_view file_prefix = "xla_compile_cache");

 private:
  JitCompilationListener* listener_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_TESTS_XLA_COMPILATION_CACHE_TEST_HELPER_H_
