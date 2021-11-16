/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_TEST_H_
#define TENSORFLOW_CORE_PLATFORM_TEST_H_

#include <memory>
#include <vector>

#include <gtest/gtest.h>  // IWYU pragma: export
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

// Includes gmock.h and enables the use of gmock matchers in tensorflow tests.
//
// Test including this header can use the macros EXPECT_THAT(...) and
// ASSERT_THAT(...) in combination with gmock matchers.
// Example:
//  std::vector<int> vec = Foo();
//  EXPECT_THAT(vec, ::testing::ElementsAre(1,2,3));
//  EXPECT_THAT(vec, ::testing::UnorderedElementsAre(2,3,1));
//
// For more details on gmock matchers see:
// https://github.com/google/googletest/blob/master/googlemock/docs/CheatSheet.md#matchers
//
// The advantages of using gmock matchers instead of self defined matchers are
// better error messages, more maintainable tests and more test coverage.
#if !defined(PLATFORM_GOOGLE) && !defined(PLATFORM_GOOGLE_ANDROID) && \
    !defined(PLATFORM_CHROMIUMOS)
#include <gmock/gmock-generated-matchers.h>
#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#endif
#include <gmock/gmock.h>

#define DISABLED_ON_GPU_ROCM(X) X
#if TENSORFLOW_USE_ROCM
#undef DISABLED_ON_GPU_ROCM
#define DISABLED_ON_GPU_ROCM(X) DISABLED_##X
#endif  // TENSORFLOW_USE_ROCM

namespace tensorflow {
namespace testing {

// Return a temporary directory suitable for temporary testing files.
//
// Where possible, consider using Env::LocalTempFilename over this function.
std::string TmpDir();

// Returns the path to TensorFlow in the directory containing data
// dependencies.
//
// A better alternative would be making use if
// tensorflow/core/platform/resource_loader.h:GetDataDependencyFilepath. That
// function should do the right thing both within and outside of tests allowing
// avoiding test specific APIs.
std::string TensorFlowSrcRoot();

// Return a random number generator seed to use in randomized tests.
// Returns the same value for the lifetime of the process.
int RandomSeed();

// Returns an unused port number, for use in multi-process testing.
// NOTE: This function is not thread-safe.
int PickUnusedPortOrDie();

}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_TEST_H_
