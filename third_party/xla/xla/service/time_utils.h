/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_TIME_UTILS_H_
#define XLA_SERVICE_TIME_UTILS_H_

#include <cstdint>

namespace xla {

// Convert between inclusive/exclusive start/end times.
int64_t ExclusiveToInclusiveStartTime(int64_t exclusive_time);
int64_t InclusiveToExclusiveStartTime(int64_t inclusive_time);
int64_t ExclusiveToInclusiveEndTime(int64_t exclusive_time);
int64_t InclusiveToExclusiveEndTime(int64_t inclusive_time);

}  // namespace xla

#endif  // XLA_SERVICE_TIME_UTILS_H_
