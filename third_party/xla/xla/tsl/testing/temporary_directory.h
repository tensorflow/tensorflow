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

#ifndef XLA_TSL_TESTING_TEMPORARY_DIRECTORY_H_
#define XLA_TSL_TESTING_TEMPORARY_DIRECTORY_H_

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"

namespace tsl {
namespace testing {

// Represents a temporary directory that is automatically deleted when this
// object is destroyed.
class TemporaryDirectory {
 public:
  // Creates a temporary directory unique to the given test case.
  static absl::StatusOr<TemporaryDirectory> CreateForTestcase(
      const ::testing::TestInfo& test_info);

  // Creates a temporary directory unique to the current test case. Returns an
  // error if not called from a test.
  static absl::StatusOr<TemporaryDirectory> CreateForCurrentTestcase();

  // Returns the path to the temporary directory.
  const std::string& path() const { return *path_; }

 private:
  explicit TemporaryDirectory(std::string path)
      : path_(new std::string(std::move(path)), RecursiveFilepathDeleter()) {}

  // A custom deleter that deletes the given directory recursively.
  struct RecursiveFilepathDeleter {
    void operator()(std::string* path) const;
  };

  // We use a unique_ptr here to get move-only semantics without
  // having to implement the move constructor and assignment operator. This of
  // course introduces a double indirection, but this is a test-only class and
  // it significantly simplifies the implementation.
  std::unique_ptr<std::string, RecursiveFilepathDeleter> path_;
};

}  // namespace testing
}  // namespace tsl

#endif  // XLA_TSL_TESTING_TEMPORARY_DIRECTORY_H_
