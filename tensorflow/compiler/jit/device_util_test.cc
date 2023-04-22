/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/device_util.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

Status PickDeviceHelper(bool allow_mixing_unknown_and_cpu,
                        absl::Span<const absl::string_view> device_names,
                        string* result) {
  jit::DeviceInfoCache cache;
  jit::DeviceSet device_set;
  for (absl::string_view name : device_names) {
    TF_ASSIGN_OR_RETURN(jit::DeviceId device_id, cache.GetIdFor(name));
    device_set.Insert(device_id);
  }

  TF_ASSIGN_OR_RETURN(
      jit::DeviceId result_id,
      PickDeviceForXla(cache, device_set, allow_mixing_unknown_and_cpu));
  *result = string(cache.GetNameFor(result_id));
  return Status::OK();
}

void CheckPickDeviceResult(absl::string_view expected_result,
                           bool allow_mixing_unknown_and_cpu,
                           absl::Span<const absl::string_view> inputs) {
  string result;
  TF_ASSERT_OK(PickDeviceHelper(allow_mixing_unknown_and_cpu, inputs, &result))
      << "inputs = [" << absl::StrJoin(inputs, ", ")
      << "], allow_mixing_unknown_and_cpu=" << allow_mixing_unknown_and_cpu
      << ", expected_result=" << expected_result;
  EXPECT_EQ(result, expected_result);
}

void CheckPickDeviceHasError(bool allow_mixing_unknown_and_cpu,
                             absl::Span<const absl::string_view> inputs) {
  string result;
  EXPECT_FALSE(
      PickDeviceHelper(allow_mixing_unknown_and_cpu, inputs, &result).ok());
}

const char* kCPU0 = "/job:localhost/replica:0/task:0/device:CPU:0";
const char* kGPU0 = "/job:localhost/replica:0/task:0/device:GPU:0";
const char* kXPU0 = "/job:localhost/replica:0/task:0/device:XPU:0";
const char* kYPU0 = "/job:localhost/replica:0/task:0/device:YPU:0";

const char* kCPU1 = "/job:localhost/replica:0/task:0/device:CPU:1";
const char* kGPU1 = "/job:localhost/replica:0/task:0/device:GPU:1";
const char* kXPU1 = "/job:localhost/replica:0/task:0/device:XPU:1";

const char* kCPU0Partial = "/device:CPU:0";
const char* kGPU0Partial = "/device:GPU:0";
const char* kXPU0Partial = "/device:XPU:0";

TEST(PickDeviceForXla, UniqueDevice) {
  CheckPickDeviceResult(kGPU0, false, {kGPU0, kGPU0});
}

TEST(PickDeviceForXla, MoreSpecificDevice) {
  CheckPickDeviceResult(kCPU0, false, {kCPU0, kCPU0Partial});
  CheckPickDeviceResult(kGPU0, false, {kGPU0, kGPU0Partial});
  // Unknown devices do not support merging of full and partial specifications.
  CheckPickDeviceHasError(false, {kXPU1, kXPU0Partial});
}

TEST(PickDeviceForXla, DeviceOrder) {
  CheckPickDeviceResult(kGPU0, false, {kGPU0, kCPU0});
  CheckPickDeviceResult(kGPU0, false, {kCPU0, kGPU0});
  CheckPickDeviceResult(kXPU0, true, {kXPU0, kCPU0});
}

TEST(PickDeviceForXla, MultipleUnknownDevices) {
  CheckPickDeviceHasError(false, {kXPU0, kYPU0});
}

TEST(PickDeviceForXla, GpuAndUnknown) {
  CheckPickDeviceHasError(false, {kGPU0, kXPU1});
}

TEST(PickDeviceForXla, UnknownAndCpu) {
  CheckPickDeviceHasError(false, {kXPU0, kCPU1});
}

TEST(PickDeviceForXla, MultipleDevicesOfSameType) {
  CheckPickDeviceHasError(true, {kCPU0, kCPU1});
  CheckPickDeviceHasError(false, {kCPU0, kCPU1});
  CheckPickDeviceHasError(false, {kGPU0, kGPU1});
  CheckPickDeviceHasError(false, {kXPU0, kXPU1});
  CheckPickDeviceHasError(false, {kCPU0, kCPU1, kGPU0});
}

void SimpleRoundTripTestForDeviceSet(int num_devices) {
  jit::DeviceSet device_set;
  jit::DeviceInfoCache device_info_cache;

  std::vector<string> expected_devices, actual_devices;

  for (int i = 0; i < num_devices; i++) {
    string device_name =
        absl::StrCat("/job:localhost/replica:0/task:0/device:XPU:", i);
    TF_ASSERT_OK_AND_ASSIGN(jit::DeviceId device_id,
                            device_info_cache.GetIdFor(device_name));
    device_set.Insert(device_id);
    expected_devices.push_back(device_name);
  }

  device_set.ForEach([&](jit::DeviceId device_id) {
    actual_devices.push_back(string(device_info_cache.GetNameFor(device_id)));
    return true;
  });

  EXPECT_EQ(expected_devices, actual_devices);
}

TEST(DeviceSetTest, SimpleRoundTrip_One) { SimpleRoundTripTestForDeviceSet(1); }

TEST(DeviceSetTest, SimpleRoundTrip_Small) {
  SimpleRoundTripTestForDeviceSet(8);
}

TEST(DeviceSetTest, SimpleRoundTrip_Large) {
  SimpleRoundTripTestForDeviceSet(800);
}

}  // namespace
}  // namespace tensorflow
