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

#ifndef TENSORFLOW_C_EAGER_C_API_EXPERIMENTAL_READER_H_
#define TENSORFLOW_C_EAGER_C_API_EXPERIMENTAL_READER_H_

#include "tensorflow/c/eager/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Test only exports of the monitoring Cell Reader API which allows tests to
// read current values from streamz counters defined in other modules.
//
// The code under test will have created streamz counters like this:
// auto* streamz = tensorflow::monitoring::Counter<1>::New("name",
// "description", "label");
// and then incremented that counter for various values of label:
// streamz->GetCell("label-value")->IncrementBy(1);
//
// The test code can then read and test the value of that counter:
//
// auto* reader = TFE_MonitoringNewCounterReader("name");
// test();
// int64_t value = TFE_MonitoringReadCounter1(reader, "label-value");

// Opaque handle to a reader.
typedef struct TFE_MonitoringCounterReader TFE_MonitoringCounterReader;

// Returns a handle to be used for reading values from streamz counter. The
// counter can have been created with any number of labels.
TF_CAPI_EXPORT extern TFE_MonitoringCounterReader*
TFE_MonitoringNewCounterReader(const char* name);

// Reads the value of a counter that was created with 0 labels.
TF_CAPI_EXPORT extern int64_t TFE_MonitoringReadCounter0(
    TFE_MonitoringCounterReader*);

// Reads the value of specific cell of a counter that was created with 1 label.
TF_CAPI_EXPORT extern int64_t TFE_MonitoringReadCounter1(
    TFE_MonitoringCounterReader*, const char* label_value);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_EAGER_C_API_EXPERIMENTAL_READER_H_
