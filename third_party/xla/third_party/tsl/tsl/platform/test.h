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

#ifndef TENSORFLOW_TSL_PLATFORM_TEST_H_
#define TENSORFLOW_TSL_PLATFORM_TEST_H_

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>  // IWYU pragma: export
#include "tsl/platform/macros.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/types.h"

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
#include <gmock/gmock-actions.h>
#include <gmock/gmock-matchers.h>            // IWYU pragma: export
#include <gmock/gmock-more-matchers.h>       // IWYU pragma: export
#endif
#include <gmock/gmock.h>  // IWYU pragma: export

namespace tsl {
namespace testing {

// Return a temporary directory suitable for temporary testing files.
//
// Where possible, consider using Env::LocalTempFilename over this function.
std::string TmpDir();

// Returns the path to TensorFlow in the directory containing data
// dependencies.
//
// A better alternative would be making use if
// tensorflow/tsl/platform/resource_loader.h:GetDataDependencyFilepath. That
// function should do the right thing both within and outside of tests allowing
// avoiding test specific APIs.
std::string TensorFlowSrcRoot();

// Returns the path to XLA in the directory containing data
// dependencies.
std::string XlaSrcRoot();

// Returns the path to TSL in the directory containing data
// dependencies.
std::string TslSrcRoot();

// Return a random number generator seed to use in randomized tests.
// Returns the same value for the lifetime of the process.
int RandomSeed();

// Returns an unused port number, for use in multi-process testing.
// NOTE: This function is not thread-safe.
int PickUnusedPortOrDie();

// Constant which is false internally and true in open source.
inline constexpr bool kIsOpenSource = TSL_IS_IN_OSS;

}  // namespace testing
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_TEST_H_
