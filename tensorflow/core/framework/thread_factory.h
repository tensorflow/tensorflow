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
#ifndef TENSORFLOW_CORE_FRAMEWORK_THREAD_FACTORY_H_
#define TENSORFLOW_CORE_FRAMEWORK_THREAD_FACTORY_H_

#include <functional>
#include <memory>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Thread;

// Virtual interface for an object that creates threads.
class ThreadFactory {
 public:
  virtual ~ThreadFactory() {}

  // Runs `fn` asynchronously in a different thread. `fn` may block.
  //
  // NOTE: The caller is responsible for ensuring that this `ThreadFactory`
  // outlives the returned `Thread`.
  virtual std::unique_ptr<Thread> StartThread(const string& name,
                                              std::function<void()> fn) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_THREAD_FACTORY_H_
