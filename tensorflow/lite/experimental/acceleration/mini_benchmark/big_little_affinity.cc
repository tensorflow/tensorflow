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

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>

#include "include/cpuinfo.h"

namespace tflite {
namespace acceleration {

namespace {
bool IsInOrderArch(cpuinfo_uarch arch) {
  switch (arch) {
    case cpuinfo_uarch_cortex_a53:
    case cpuinfo_uarch_cortex_a55r0:
    case cpuinfo_uarch_cortex_a55:
    case cpuinfo_uarch_cortex_a57:
      return true;
    default:
      return false;
  }
  return false;
}
}  // namespace

BigLittleAffinity GetAffinity() {
  BigLittleAffinity affinity;
  if (!cpuinfo_initialize()) {
    return affinity;
  }
  std::map<uint32_t, uint64_t> cluster_to_max_frequency;
  uint64_t smallest_max_frequency = UINT64_MAX;
  uint64_t largest_max_frequency = 0;
  uint64_t processors_count = cpuinfo_get_processors_count();
  for (auto i = 0; i < processors_count; i++) {
    const struct cpuinfo_processor* processor = cpuinfo_get_processor(i);
    if (processor->core->frequency > 0) {
      cluster_to_max_frequency[processor->cluster->cluster_id] =
          processor->core->frequency;
      smallest_max_frequency =
          std::min(smallest_max_frequency, processor->core->frequency);
      largest_max_frequency =
          std::max(largest_max_frequency, processor->core->frequency);
    }
  }

  int count_of_processors_with_largest_max_frequency = 0;
  for (auto i = 0; i < cpuinfo_get_processors_count(); i++) {
    const struct cpuinfo_processor* processor = cpuinfo_get_processor(i);
    uint64_t max_frequency =
        cluster_to_max_frequency[processor->cluster->cluster_id];
    if (max_frequency == largest_max_frequency) {
      ++count_of_processors_with_largest_max_frequency;
    }
  }
  std::set<cpuinfo_uarch> archs;

  // Three variants for detecting the big/little split:
  // - all cores have the same frequency, check the uarch for in-order (on
  //   big.LITTLE, the big cores are typically out-of-order and the LITTLE
  //   cores in-order)
  // - if there are 2 cores with largest max frequency, those are counted as big
  // - otherwise the cores with smallest max frequency are counted as LITTLE
  for (auto i = 0; i < cpuinfo_get_processors_count(); i++) {
    const struct cpuinfo_processor* processor = cpuinfo_get_processor(i);
    uint64_t max_frequency =
        cluster_to_max_frequency[processor->cluster->cluster_id];
    bool is_little;
    archs.insert(processor->core->uarch);
    if (count_of_processors_with_largest_max_frequency ==
        cpuinfo_get_processors_count()) {
      is_little = IsInOrderArch(processor->core->uarch);
    } else if (count_of_processors_with_largest_max_frequency == 2) {
      is_little = (max_frequency != largest_max_frequency);
    } else {
      is_little = (max_frequency == smallest_max_frequency);
    }
#ifdef __ANDROID__
    // On desktop linux there are easily more processors than bits in an int, so
    // skip this code. It's still convenient to enable the rest of the code on
    // non-Android for quicker testing.
    if (is_little) {
      affinity.little_core_affinity |= (0x1 << processor->linux_id);
    } else {
      affinity.big_core_affinity |= (0x1 << processor->linux_id);
    }
#endif  // __ANDROID__
  }
  // After the detection we may have determined that all cores are big or
  // LITTLE. This is ok if there is only one cluster or if all the cores are the
  // same, and in that case we return the same for both masks.
  if (cluster_to_max_frequency.size() == 1) {
    // Only one cluster.
    affinity.big_core_affinity = affinity.little_core_affinity =
        std::max(affinity.big_core_affinity, affinity.little_core_affinity);
  } else if (count_of_processors_with_largest_max_frequency ==
                 cpuinfo_get_processors_count() &&
             archs.size() == 1) {
    // All cores have same uarch and frequency.
    affinity.big_core_affinity = affinity.little_core_affinity =
        std::max(affinity.big_core_affinity, affinity.little_core_affinity);
  }
  return affinity;
}

}  // namespace acceleration
}  // namespace tflite
