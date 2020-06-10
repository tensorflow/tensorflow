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
#include "tensorflow/lite/experimental/acceleration/compatibility/devicedb.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/compatibility/database_generated.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/devicedb-sample.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/variables.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace acceleration {
namespace {

class DeviceDbTest : public ::testing::Test {
 protected:
  void LoadSample() {
    device_db_ = flatbuffers::GetRoot<DeviceDatabase>(
        g_tflite_acceleration_devicedb_sample_binary);
  }

  const DeviceDatabase* device_db_ = nullptr;
};

TEST_F(DeviceDbTest, Load) {
  LoadSample();
  ASSERT_TRUE(device_db_);
  ASSERT_TRUE(device_db_->root());
  EXPECT_EQ(device_db_->root()->size(), 3);
}

TEST_F(DeviceDbTest, SocLookup) {
  LoadSample();
  ASSERT_TRUE(device_db_);
  std::map<std::string, std::string> variables;

  // Find first device mapping.
  variables[kDeviceModel] = "m712c";
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables[kSoCModel], "exynos_7872");

  // Find second device mapping.
  variables.clear();
  variables[kDeviceModel] = "sc_02l";
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables[kSoCModel], "exynos_7885");

  // Make sure no results are returned without a match.
  variables.clear();
  variables[kDeviceModel] = "nosuch";
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables.find(kSoCModel), variables.end());
}

TEST_F(DeviceDbTest, StatusLookupWithSoC) {
  LoadSample();
  ASSERT_TRUE(device_db_);
  std::map<std::string, std::string> variables;

  // Find exact match.
  variables[kOpenGLESVersion] = "3.1";
  variables[kSoCModel] = "exynos_7872";
  variables[kAndroidSdkVersion] = "24";
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables[gpu::kStatus], gpu::kStatusSupported);

  // Ensure no results without a match.
  variables[kOpenGLESVersion] = "3.0";
  variables.erase(variables.find(gpu::kStatus));
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables.find(gpu::kStatus), variables.end());

  // Find no results with too low an android version.
  variables.clear();
  variables[kOpenGLESVersion] = "3.1";
  variables[kSoCModel] = "exynos_7883";
  variables[kAndroidSdkVersion] = "24";
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables.find(gpu::kStatus), variables.end());
  // Find a match with android version above minimum.
  variables[kAndroidSdkVersion] = "29";
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables[gpu::kStatus], gpu::kStatusSupported);
}

TEST_F(DeviceDbTest, StatusLookupWithDevice) {
  LoadSample();
  ASSERT_TRUE(device_db_);
  std::map<std::string, std::string> variables;
  // Find unsupported device (same model, different device).
  variables[kAndroidSdkVersion] = "24";
  variables[kDeviceModel] = "sm_j810f";
  variables[kDeviceName] = "j8y18lte";
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables[gpu::kStatus], gpu::kStatusUnsupported);

  // Find supported device (same model, different device).
  variables.clear();
  variables[kAndroidSdkVersion] = "24";
  variables[kDeviceModel] = "sm_j810m";
  variables[kDeviceName] = "j8y18lte";
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables[gpu::kOpenCLStatus], gpu::kStatusSupported);
}

TEST_F(DeviceDbTest, StatusLookupBasedOnDerivedProperties) {
  LoadSample();
  ASSERT_TRUE(device_db_);
  std::map<std::string, std::string> variables;
  // Find status based on SoC derived from model.
  variables[kOpenGLESVersion] = "3.1";
  variables[kAndroidSdkVersion] = "24";
  variables[kDeviceModel] = "m712c";
  UpdateVariablesFromDatabase(&variables, *device_db_);
  EXPECT_EQ(variables[gpu::kStatus], gpu::kStatusSupported);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
