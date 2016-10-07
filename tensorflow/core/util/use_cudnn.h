/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// The utility to check whether we have Cudnn dependency.

#ifndef TENSORFLOW_UTIL_USE_CUDNN_H_
#define TENSORFLOW_UTIL_USE_CUDNN_H_

namespace tensorflow {

bool CanUseCudnn();
bool CudnnUseAutotune();

namespace internal {

// This function is for transition only. And it may go away at any time.
bool AvgPoolUseCudnn();

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_USE_CUDNN_H_
