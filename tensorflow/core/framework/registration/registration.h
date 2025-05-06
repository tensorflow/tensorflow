/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/registration/registration.h"

#include <iostream>  // Added for logging output
#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::testing::Eq;

#define STORE_NEXT_ID_IMPL(id, name) constexpr int name = id
#define STORE_NEXT_ID(name) TF_NEW_ID_FOR_INIT(STORE_NEXT_ID_IMPL, name)

// Generate compile-time unique IDs
STORE_NEXT_ID(kBaseId);
STORE_NEXT_ID(kNextId1);
STORE_NEXT_ID(kNextId2);

TEST(NewIdForInitTest, SequentialIds) {
  static_assert(kBaseId >= 0, "kBaseId < 0");
  static_assert(kNextId1 == kBaseId + 1, "kNextId1 != kBaseId+1");
  static_assert(kNextId2 == kBaseId + 2, "kNextId2 != kBaseId+2");
}

// Static initialization test (unconditional)
int observed_unconditional_init = 0;
InitOnStartupMarker const kUnconditionalInitMarker =
    InitOnStartupMarker{} << []() {
      std::cout << "Running unconditional init.\n";
      observed_unconditional_init++;
      return InitOnStartupMarker{};
    };

TEST(InitOnStartupTest, Unconditional) {
  EXPECT_THAT(observed_unconditional_init, Eq(1));
}

// Conditional initialization
template <bool Enable>
int observed_conditional_init = 0;

template <bool Enable>
InitOnStartupMarker const kConditionalInitMarker =
    TF_INIT_ON_STARTUP_IF(Enable) << []() {
      std::cout << "Running conditional init (Enable=" << Enable << ").\n";
      (observed_conditional_init<Enable>)++;
      return InitOnStartupMarker{};
    };

// Explicit template instantiation
template InitOnStartupMarker const kConditionalInitMarker<true>;
template InitOnStartupMarker const kConditionalInitMarker<false>;

TEST(InitOnStartupTest, Conditional) {
  EXPECT_THAT(observed_conditional_init<true>, Eq(1));
  EXPECT_THAT(observed_conditional_init<false>, Eq(0));
}

// Conditional immediate initialization
template <bool Enable>
int observed_conditional_init_immediate = 0;

template <bool Enable>
InitOnStartupMarker const kConditionalInitImmediateMarker =
    TF_INIT_ON_STARTUP_IF(Enable) << ([]() {
      std::cout << "Running conditional immediate init (Enable=" << Enable << ").\n";
      (observed_conditional_init_immediate<Enable>)++;
      return InitOnStartupMarker{};
    })();

template InitOnStartupMarker const kConditionalInitImmediateMarker<true>;
template InitOnStartupMarker const kConditionalInitImmediateMarker<false>;

TEST(InitOnStartupTest, ConditionalImmediate) {
  EXPECT_THAT(observed_conditional_init_immediate<true>, Eq(1));
  EXPECT_THAT(observed_conditional_init_immediate<false>, Eq(0));
}

}  // namespace
}  // namespace tensorflow
