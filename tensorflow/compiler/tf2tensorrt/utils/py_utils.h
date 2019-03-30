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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_PY_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_PY_UTILS_H_

namespace tensorflow {
namespace tensorrt {

bool IsGoogleTensorRTEnabled();

// Return compile time TensorRT library version information {Maj, Min, Patch}.
void GetLinkedTensorRTVersion(int* major, int* minor, int* patch);

// Return runtime time TensorRT library version information {Maj, Min, Patch}.
void GetLoadedTensorRTVersion(int* major, int* minor, int* patch);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_PY_UTILS_H_
