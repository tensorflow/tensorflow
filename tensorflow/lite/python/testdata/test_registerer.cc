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
#include "tensorflow/lite/python/testdata/test_registerer.h"

namespace tflite {

namespace {
static int num_test_registerer_calls = 0;
}  // namespace

// Dummy registerer function with the correct signature. Ignores the resolver
// but increments the num_test_registerer_calls counter by one. The TF_ prefix
// is needed to get past the version script in the OSS build.
extern "C" void TF_TestRegisterer(tflite::MutableOpResolver *resolver) {
  num_test_registerer_calls++;
}

// Returns the num_test_registerer_calls counter and re-sets it.
int get_num_test_registerer_calls() {
  const int result = num_test_registerer_calls;
  num_test_registerer_calls = 0;
  return result;
}

}  // namespace tflite
