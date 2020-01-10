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

#ifndef TENSORFLOW_CORE_UTIL_EXEC_ON_STALL_H_
#define TENSORFLOW_CORE_UTIL_EXEC_ON_STALL_H_

#include <functional>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// An object that executes a particular function only if it
// is not deleted within the allotted number of seconds.
//
// This can be useful in diagnosing deadlocks, stalls and memory leaks
// without logging too aggressively.
class ExecuteOnStall {
 public:
  // delay_secs: If the object still exists after this many seconds,
  //     execute f.
  // f: The function to be executed, for example a detailed log of the
  //    the state of an object to which this is attached.
  // poll_microseconds: The spawned thread will wake and test whether
  //    the destructor has been invoked this frequently.
  ExecuteOnStall(int delay_secs, std::function<void()> f,
                 int32 poll_microseconds = 100)
      : disabled_(false),
        joined_(false),
        env_(Env::Default()),
        f_(f),
        poll_microseconds_(poll_microseconds) {
    deadline_ = env_->NowMicros() + 1000000 * delay_secs;
    env_->SchedClosure([this]() {
      while (env_->NowMicros() < deadline_) {
        {
          mutex_lock l(mu_);
          if (disabled_) {
            break;
          }
        }
        env_->SleepForMicroseconds(poll_microseconds_);
      }
      {
        mutex_lock l(mu_);
        if (!disabled_) {
          f_();
        }
        joined_ = true;
        cond_var_.notify_all();
      }
    });
  }

  ~ExecuteOnStall() {
    // Wait for spawned thread to terminate.
    mutex_lock l(mu_);
    disabled_ = true;
    if (!joined_) {
      cond_var_.wait(l);
    }
  }

 private:
  mutex mu_;
  condition_variable cond_var_;
  bool disabled_ GUARDED_BY(mu_);
  bool joined_ GUARDED_BY(mu_);
  Env* env_;
  std::function<void()> f_;
  int64 deadline_;
  int32 poll_microseconds_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_UTIL_EXEC_ON_STALL_H_
