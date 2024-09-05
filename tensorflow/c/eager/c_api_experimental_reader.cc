/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
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

#include "tensorflow/c/eager/tfe_monitoring_reader_internal.h"

template <typename... LabelType>
int64_t TFE_MonitoringCounterReader::Read(const LabelType&... labels) {
  return counter->Read(labels...);
}

TFE_MonitoringCounterReader* TFE_MonitoringNewCounterReader(const char* name) {
  auto* result = new TFE_MonitoringCounterReader(name);

  return result;
}

int64_t TFE_MonitoringReadCounter0(TFE_MonitoringCounterReader* cell_reader) {
  int64_t result = cell_reader->Read();

  return result;
}

int64_t TFE_MonitoringReadCounter1(TFE_MonitoringCounterReader* cell_reader,
                                   const char* label) {
  int64_t result = cell_reader->Read(label);

  return result;
}
