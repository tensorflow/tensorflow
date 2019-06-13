/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/tune.h"

#include <chrono>  // NOLINT(build/c++11)
#include <thread>  // NOLINT(build/c++11)

#include <gtest/gtest.h>

namespace ruy {
namespace {

TEST(TuneTest, TuneTest) {
  TuningResolver tuning_resolver;
  ASSERT_FALSE(tuning_resolver.Resolve() == Tuning::kAuto);
  // 1 second is likely higher than TuningResolver's internal cache expiry,
  // exercising the logic invalidating earlier tuning resolutions.
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_FALSE(tuning_resolver.Resolve() == Tuning::kAuto);

  tuning_resolver.SetTuning(Tuning::kAuto);

  for (auto tuning : {Tuning::kOutOfOrder, Tuning::kInOrder}) {
    tuning_resolver.SetTuning(tuning);
    ASSERT_TRUE(tuning_resolver.Resolve() == tuning);
    // See above comment about 1 second.
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ASSERT_TRUE(tuning_resolver.Resolve() == tuning);
  }
}

}  // namespace
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
