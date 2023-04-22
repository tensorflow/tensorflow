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

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::testing::Eq;

#define STORE_NEXT_ID_IMPL(id, name) constexpr int name = id
#define STORE_NEXT_ID(name) TF_NEW_ID_FOR_INIT(STORE_NEXT_ID_IMPL, name)

STORE_NEXT_ID(kBaseId);
STORE_NEXT_ID(kNextId1);
STORE_NEXT_ID(kNextId2);

TEST(NewIdForInitTest, SequentialIds) {
  static_assert(kBaseId >= 0, "kBaseId < 0");
  static_assert(kNextId1 == kBaseId + 1, "kNextId1 != kBaseId+1");
  static_assert(kNextId2 == kBaseId + 2, "kNextId2 != kBaseId+2");
}

int observed_unconditional_init;
InitOnStartupMarker const kUnconditionalInitMarker =
    InitOnStartupMarker{} << []() {
      observed_unconditional_init++;
      return InitOnStartupMarker{};
    };

TEST(InitOnStartupTest, Unconditional) {
  EXPECT_THAT(observed_unconditional_init, Eq(1));
}

template <bool Enable>
int observed_conditional_init;
template <bool Enable>
InitOnStartupMarker const kConditionalInitMarker =
    TF_INIT_ON_STARTUP_IF(Enable) << []() {
      (observed_conditional_init<Enable>)++;
      return InitOnStartupMarker{};
    };

template InitOnStartupMarker const kConditionalInitMarker<true>;
template InitOnStartupMarker const kConditionalInitMarker<false>;

// TODO(b/169282173): Enable once the issue is fixed.
TEST(InitOnStartupTest, DISABLED_Conditional) {
  EXPECT_THAT(observed_conditional_init<true>, Eq(1));
  EXPECT_THAT(observed_conditional_init<false>, Eq(0));
}

template <bool Enable>
int observed_conditional_init_immediate;
template <bool Enable>
InitOnStartupMarker const kConditionalInitImmediateMarker =
    TF_INIT_ON_STARTUP_IF(Enable) << ([]() {
      (observed_conditional_init_immediate<Enable>)++;
      return InitOnStartupMarker{};
    })();

template InitOnStartupMarker const kConditionalInitImmediateMarker<true>;
template InitOnStartupMarker const kConditionalInitImmediateMarker<false>;

// TODO(b/169282173): Enable once the issue is fixed.
TEST(InitOnStartupTest, DISABLED_ConditionalImmediate) {
  EXPECT_THAT(observed_conditional_init_immediate<true>, Eq(1));
  EXPECT_THAT(observed_conditional_init_immediate<false>, Eq(0));
}

}  // namespace
}  // namespace tensorflow
