/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/big_little_affinity.h"

#include <cstdint>
#include <map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/cpuinfo.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"

namespace tflite {
namespace acceleration {
namespace {

TEST(BigLittle, CheckBasics) {
  ASSERT_TRUE(cpuinfo_initialize());
  auto processors_count = cpuinfo_get_processors_count();
  ASSERT_GT(processors_count, 0);
#if defined(__ANDROID__)
  AndroidInfo android_info;
  auto status = RequestAndroidInfo(&android_info);
  if (android_info.is_emulator) {
    std::cout << "Running on emulator\n";
    return;
  } else {
    std::cout << "Running on hardware\n";
  }
  ASSERT_TRUE(status.ok());
  std::map<uint32_t, uint64_t> cluster_to_max_frequency;
  for (auto i = 0; i < cpuinfo_get_processors_count(); i++) {
    const struct cpuinfo_processor* processor = cpuinfo_get_processor(i);
    if (processor->core->frequency > 0) {
      cluster_to_max_frequency[processor->cluster->cluster_id] =
          processor->core->frequency;
    }
  }
  EXPECT_GT(cluster_to_max_frequency.size(), 0);
  EXPECT_LE(cluster_to_max_frequency.size(), 3);
  for (auto i = 0; i < cpuinfo_get_processors_count(); i++) {
    const struct cpuinfo_processor* processor = cpuinfo_get_processor(i);
    EXPECT_TRUE(cluster_to_max_frequency.find(processor->cluster->cluster_id) !=
                cluster_to_max_frequency.end());
  }
  BigLittleAffinity affinity = GetAffinity();
  EXPECT_GT(affinity.little_core_affinity, 0);
  EXPECT_GT(affinity.big_core_affinity, 0);
  std::cout << "Little core affinity: " << std::hex
            << affinity.little_core_affinity << std::endl;
  std::cout << "Big core affinity: " << std::hex << affinity.big_core_affinity
            << std::endl;
#endif
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
