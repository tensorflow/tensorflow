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
#include <string>
#include <vector>

namespace tensorflow {

// Assign thread ranges to requests.
// Requests are numbered 0...num_active_requests-1, and
// threads are numbered 0...num_threads-1.
// On return, the range [start_vec->at(i), end_vec->at(i))
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

// Assign thread steal ranges to threads.Threads are numbered 0...num_threads-1.
// On return, the range [start_vec->at(i), end_vec->at(i)) indicates the steal
// range of the thread i. The ranges given to different threads may overlap.
void ComputeInterOpStealingRanges(int num_threads, int min_threads_per_domain,
                                  std::vector<std::uint_fast32_t>* start_vec,
                                  std::vector<std::uint_fast32_t>* end_vec);

// For each of the num_threads determine the index of the active_request whose
// work queue should be attempted first by that the thread. Return a vector of
// size num_threads which represents how threads should be distributed across
// requests.
std::vector<int> ChooseRequestsWithExponentialDistribution(
    int num_active_requests, int num_threads);

// Look up environment variable named 'var_name' and return the value if it
// exist and can be parsed. Return 'default_value' otherwise.
double ParamFromEnvWithDefault(const char* var_name, double default_value);

// Look up environment variable named 'var_name' and return the value if it
// exist and can be parsed. The value must be in format val1,val2... Return
// 'default_value' otherwise.
std::vector<double> ParamFromEnvWithDefault(const char* var_name,
                                            std::vector<double> default_value);

// Look up environment variable named 'var_name' and return the value if it
// exist and can be parsed. The value must be in format val1,val2... Return
// 'default_value' otherwise.
std::vector<int> ParamFromEnvWithDefault(const char* var_name,
                                         std::vector<int> default_value);

// Look up environment variable named 'var_name' and return the value if it
// exist and can be parsed. Return 'default_value' otherwise.
bool ParamFromEnvBoolWithDefault(const char* var_name, bool default_value);

}  // end namespace tensorflow
#endif  // TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_UTIL_H_
