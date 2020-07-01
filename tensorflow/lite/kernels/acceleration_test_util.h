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
#ifndef TENSORFLOW_LITE_KERNELS_ACCELERATION_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_ACCELERATION_TEST_UTIL_H_

#include <string>

namespace tflite {

// Returns the test id to use to retrieve the acceleration configuration
// in the acceleration whitelist.
std::string GetCurrentTestId();

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_ACCELERATION_TEST_UTIL_H_
