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
#include "tensorflow/core/data/utils.h"

#include <string>

#include "tensorflow/core/framework/metrics.h"

namespace tensorflow {
namespace data {

void AddLatencySample(int64_t microseconds) {
  metrics::RecordTFDataGetNextDuration(microseconds);
}

void IncrementThroughput(int64_t bytes) {
  metrics::RecordTFDataBytesFetched(bytes);
}

std::string TranslateFileName(const std::string& fname) { return fname; }

}  // namespace data
}  // namespace tensorflow
