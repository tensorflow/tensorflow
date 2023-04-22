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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_REQUEST_ID_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_REQUEST_ID_H_

#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Returns a request_id for use with RecentRequestIds. This number will not be
// zero, and must be unique over RecentRequestIds' window of
// num_tracked_request_ids. See recent_request_ids.h for more details.
int64 GetUniqueRequestId();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_REQUEST_ID_H_
