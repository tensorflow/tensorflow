/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/test.h"

class LocalClientAotTest : public ::testing::Test {};

// This is a compiled XLA computation which calls SumStructElements, and then
// doubles the result.
extern "C" void SumAndDouble(float* out, xla::ExecutableRunOptions* options,
                             void** parameters, void** temporary_buffers);

// Just some structs with some arbitrary fields used to test the OPAQUE type.
struct OpaqueData {
  int field1 : 15;
  int field2 : 14;
  int field3 : 3;
};

// This is the implementation of a custom op which will be called by
// SumAndDouble.
extern "C" void SumStructElements(float* out, void** parameters) {
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(parameters, sizeof(OpaqueData*));
  const auto* opaque_data = static_cast<OpaqueData*>(parameters[0]);
  *out = opaque_data->field1 + opaque_data->field2 + opaque_data->field3;
}

TEST_F(LocalClientAotTest, Constant) {
  xla::ExecutableRunOptions run_options;
  OpaqueData opaque_data{100, 20, 3};
  void* parameters[] = {&opaque_data};
  float out = 0;
  char tmp[4] = {0};
  void* temporary_buffers[] = {nullptr, &out, &tmp};
  SumAndDouble(&out, &run_options, parameters, temporary_buffers);
  EXPECT_EQ(out, 246.0f);

  opaque_data = {1, 2, 3};
  SumAndDouble(&out, &run_options, parameters, temporary_buffers);
  EXPECT_EQ(out, 12.0f);
}
