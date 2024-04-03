/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/array.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/mock.h"

namespace xla {
namespace ifrt {
namespace {

TEST(ArrayTest, MakeArrayPointerListTest) {
  const int kNumArrays = 3;
  std::vector<tsl::RCReference<Array>> arrays;
  arrays.reserve(kNumArrays);
  for (int i = 0; i < kNumArrays; ++i) {
    arrays.push_back(tsl::MakeRef<MockArray>());
  }

  std::vector<Array*> array_pointer_list = MakeArrayPointerList(arrays);
  ASSERT_THAT(array_pointer_list, testing::SizeIs(kNumArrays));
  for (int i = 0; i < kNumArrays; ++i) {
    EXPECT_THAT(array_pointer_list[i], arrays[i].get());
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
