/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_UTILS_H_
#define TENSORFLOW_CORE_DATA_UTILS_H_

#include <string>

namespace tensorflow {
namespace data {

// Records latency of fetching data from tf.data iterator.
void AddLatencySample(int64_t microseconds);

// Records bytes produced by a tf.data iterator.
void IncrementThroughput(int64_t bytes);

// Returns a modified file name that can be used to do implementation specific
// file name manipulation/optimization.
std::string TranslateFileName(const std::string& fname);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_UTILS_H_
