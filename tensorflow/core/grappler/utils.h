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

#ifndef TENSORFLOW_GRAPPLER_UTILS_H_
#define TENSORFLOW_GRAPPLER_UTILS_H_

#include <functional>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

// Return the node name corresponding to 'name' if name is valid, or the empty
// string otherwise.
string NodeName(const string& name);

// Get the trailing position number ":{digits}" (if any) of a node name.
int NodePosition(const string& name);

// Add a prefix to a node name
string AddPrefixToNodeName(const string& name, const string& prefix);

// Executes a 'fn' in the 'thread_pool'. The method waits for the configured
// timeout (in milliseconds) for 'fn' to complete, before returning false.
//
// If returning false, the 'fn' may still continue to execute in the
// thread-pool. It is the responsibility of the caller to reset the thread-pool
// as appropriate.
bool ExecuteWithTimeout(std::function<void()> fn, int64 timeout_in_ms,
                        thread::ThreadPool* thread_pool);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_UTILS_H_
