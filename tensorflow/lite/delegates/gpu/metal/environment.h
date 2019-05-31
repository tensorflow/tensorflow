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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_ENVIRONMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_ENVIRONMENT_H_

namespace tflite {
namespace gpu {
namespace metal {

// Returns runtime operation system version. Example 10.1
float GetiOsSystemVersion();

// Returns Apple SoC generation number. The list of Apple SoC that support Metal
// API:
//     7 - A7 iPhone 5s, iPad Air, iPad Mini 2, iPad Mini 3.
//     8 - A8 iPhone 6, A8X iPad Air 2, iPad Mini 4.
//     9 - A9 iPhone 6s, iPad (2017), A9X iPad Pro (1st generation).
//     10 - A10 iPhone 7, iPad (2018), A10X iPad Pro (2nd generation).
//     11 - A11 iPhone 8/X.
//     12 - A12 iPhone Xs.
int GetAppleSocVersion();

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_ENVIRONMENT_H_
