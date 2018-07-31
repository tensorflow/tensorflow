/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_CONVERT_UTILS_H_
#define TENSORFLOW_CONTRIB_TENSORRT_CONVERT_UTILS_H_

#include <memory>

namespace tensorflow {
namespace tensorrt {

template <typename T>
struct TrtDestroyer {
  void operator()(T* t) {
    if (t) t->destroy();
  }
};

template <typename T>
using TrtUniquePtrType = std::unique_ptr<T, TrtDestroyer<T>>;

bool IsGoogleTensorRTEnabled();

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSORRT_CONVERT_UTILS_H_
