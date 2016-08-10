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

#ifndef TENSORFLOW_LIB_CORE_BLOCKING_COUNTER_H_
#define TENSORFLOW_LIB_CORE_BLOCKING_COUNTER_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class BlockingCounter {
 public:
  BlockingCounter(int initial_count) : count_(initial_count) {
    CHECK_GE(count_, 0);
  }

  ~BlockingCounter() {}

  inline void DecrementCount() {
    mutex_lock l(mu_);
    --count_;
    CHECK(count_ >= 0);
    if (count_ == 0) {
      cond_var_.notify_all();
    }
  }

  inline void Wait() {
    mutex_lock l(mu_);
    while (count_ > 0) {
      cond_var_.wait(l);
    }
  }

 private:
  int count_;
  mutex mu_;
  condition_variable cond_var_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_BLOCKING_COUNTER_H_
