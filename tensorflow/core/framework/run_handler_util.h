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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_UTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_UTIL_H_

#include <cstdint>
#include <vector>

namespace tensorflow {

// Assign thread ranges to requests.
// Requests are numbered 0...num_active_requests-1, and
// threads are numbered 0...num_threads-1.
// On return, the range start_vec->at(i)...end_vec->at(i)-1
// indicates the subrange of the threads available to request i.
// The ranges given to different requests may overlap.
// Lower numbered requests will tend to be assigned more threads.
// Thus, a client might associate older requests with lower
// array indices so they receive access to more threads.
// However, the routine ensures that each request is given access
// to at least min(min_threads_per_request, num_threads)  threads.
// Every thread will be assigned to at least one request range,
// assuming there is at least one request.
void ComputeInterOpSchedulingRanges(int num_active_requests, int num_threads,
                                    int min_threads_per_request,
                                    std::vector<std::uint_fast32_t>* start_vec,
                                    std::vector<std::uint_fast32_t>* end_vec);

}  // end namespace tensorflow
#endif  // TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_UTIL_H_
