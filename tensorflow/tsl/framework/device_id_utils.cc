/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/framework/device_id_utils.h"

#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "tensorflow/tsl/framework/device_id.h"
#include "tensorflow/tsl/framework/device_id_manager.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/str_util.h"

namespace tsl {

void CheckValidTfDeviceId(const DeviceType& type,
                          const int visible_device_count,
                          const TfDeviceId tf_device_id) {
  PlatformDeviceId platform_device_id;
  TF_CHECK_OK(DeviceIdManager::TfToPlatformDeviceId(type, tf_device_id,
                                                    &platform_device_id));
  CHECK_LT(platform_device_id.value(), visible_device_count)  // Crash OK
      << "platform_device_id is outside discovered device range."
      << " TF " << type << " id: " << tf_device_id << ", platform " << type
      << " id: " << platform_device_id
      << ", visible device count: " << visible_device_count;
}

Status ParseVisibleDeviceList(
    const std::string& visible_device_list, const int visible_device_count,
    std::vector<PlatformDeviceId>* visible_device_order) {
  visible_device_order->clear();

  // If the user wants to remap the visible to virtual Device mapping,
  // check for that here.
  if (visible_device_list.empty()) {
    visible_device_order->resize(visible_device_count);
    // By default, visible to virtual mapping is unchanged.
    std::iota(visible_device_order->begin(), visible_device_order->end(), 0);
  } else {
    const std::vector<std::string> order_str =
        tsl::str_util::Split(visible_device_list, ',');
    for (const std::string& platform_device_id_str : order_str) {
      int32_t platform_device_id;
      if (!absl::SimpleAtoi(platform_device_id_str, &platform_device_id)) {
        return tsl::errors::InvalidArgument(
            "Could not parse entry in 'visible_device_list': '",
            platform_device_id_str,
            "'. visible_device_list = ", visible_device_list);
      }
      if (platform_device_id < 0 ||
          platform_device_id >= visible_device_count) {
        return tsl::errors::InvalidArgument(
            "'visible_device_list' listed an invalid Device id '",
            platform_device_id, "' but visible device count is ",
            visible_device_count);
      }
      visible_device_order->push_back(
          tsl::PlatformDeviceId(platform_device_id));
    }
  }

  // Validate no repeats.
  std::set<PlatformDeviceId> visible_device_set(visible_device_order->begin(),
                                                visible_device_order->end());
  if (visible_device_set.size() != visible_device_order->size()) {
    return tsl::errors::InvalidArgument(
        "visible_device_list contained a duplicate entry: ",
        visible_device_list);
  }
  return tsl::OkStatus();
}

}  // namespace tsl
