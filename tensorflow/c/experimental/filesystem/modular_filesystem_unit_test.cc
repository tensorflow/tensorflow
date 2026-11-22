/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Unit tests for `ModularFileSystem` that don't need plugin DSOs. See
// modular_filesystem_test.cc for the plugin compliance tests.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/experimental/filesystem/modular_filesystem.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace {

using ::absl_testing::IsOk;
using ::testing::ElementsAre;

// A plugin `get_matching_paths` that returns its matches out of order.
int GetMatchingPathsOutOfOrder(const TF_Filesystem* /*filesystem*/,
                               const char* /*glob*/, char*** entries,
                               TF_Status* /*status*/) {
  const std::vector<std::string> matches = {"c", "a", "b"};
  *entries = static_cast<char**>(std::malloc(matches.size() * sizeof(char*)));
  for (size_t i = 0; i < matches.size(); ++i) {
    const size_t size = matches[i].size() + 1;
    (*entries)[i] = static_cast<char*>(std::malloc(size));
    std::memcpy((*entries)[i], matches[i].c_str(), size);
  }
  return static_cast<int>(matches.size());
}

TEST(ModularFileSystemUnitTest, GetMatchingPathsSortsPluginResults) {
  auto ops = std::make_unique<TF_FilesystemOps>();
  ops->cleanup = [](TF_Filesystem*) {};
  ops->get_matching_paths = &GetMatchingPathsOutOfOrder;
  ModularFileSystem fs(
      std::make_unique<TF_Filesystem>(), std::move(ops),
      /*random_access_file_ops=*/nullptr, /*writable_file_ops=*/nullptr,
      /*read_only_memory_region_ops=*/nullptr,
      /*plugin_memory_allocate=*/[](size_t size) { return std::malloc(size); },
      /*plugin_memory_free=*/[](void* ptr) { std::free(ptr); });

  std::vector<std::string> results;
  ASSERT_THAT(fs.GetMatchingPaths("*", &results), IsOk());
  EXPECT_THAT(results, ElementsAre("a", "b", "c"));
}

}  // namespace
}  // namespace tensorflow
