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

#include "tensorflow/c/eager/c_api_experimental_reader.h"

#include <cstdint>

#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TFE_MonitoringCounter0* CreateCounter0(const char* counter_name);
TFE_MonitoringCounter1* CreateCounter1(const char* counter_name,
                                       const char* label);
void IncrementCounter0(TFE_MonitoringCounter0* counter, int64_t delta = 1);
void IncrementCounter1(TFE_MonitoringCounter1* counter, const char* label,
                       int64_t delta = 1);

TEST(CAPI, MonitoringCellReader0) {
  auto counter_name = "test/counter0";
  auto* counter = CreateCounter0(counter_name);
  auto* reader = TFE_MonitoringNewCounterReader(counter_name);
  IncrementCounter0(counter);

  int64_t actual = TFE_MonitoringReadCounter0(reader);

  CHECK_EQ(actual, 1);
}

TEST(CAPI, MonitoringCellReader1) {
  auto counter_name = "test/counter1";
  auto label_name = "test/label";
  auto* counter = CreateCounter1(counter_name, label_name);
  auto* reader = TFE_MonitoringNewCounterReader(counter_name);
  IncrementCounter1(counter, label_name);

  int64_t actual = TFE_MonitoringReadCounter1(reader, label_name);

  CHECK_EQ(actual, 1);
}

TFE_MonitoringCounter0* CreateCounter0(const char* counter_name) {
  TF_Status* status = TF_NewStatus();
  auto* counter =
      TFE_MonitoringNewCounter0(counter_name, status, "description");
  TF_DeleteStatus(status);
  return counter;
}

void IncrementCounter0(TFE_MonitoringCounter0* counter, int64_t delta) {
  auto* cell = TFE_MonitoringGetCellCounter0(counter);
  TFE_MonitoringCounterCellIncrementBy(cell, delta);
}

TFE_MonitoringCounter1* CreateCounter1(const char* counter_name,
                                       const char* label) {
  TF_Status* status = TF_NewStatus();
  auto* counter =
      TFE_MonitoringNewCounter1(counter_name, status, "description", label);
  TF_DeleteStatus(status);
  return counter;
}

void IncrementCounter1(TFE_MonitoringCounter1* counter, const char* label,
                       int64_t delta) {
  auto* cell = TFE_MonitoringGetCellCounter1(counter, label);
  TFE_MonitoringCounterCellIncrementBy(cell, delta);
}

}  // namespace
}  // namespace tensorflow
