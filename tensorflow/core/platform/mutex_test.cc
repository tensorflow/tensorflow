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

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Check that mutex_lock and shared_mutex_lock are movable and that their
// thread-safety annotations are correct enough that we don't get an error when
// we use a moved-from lock.  (For instance, we might incorrectly get an error
// at the end of Test() when we destruct the mutex_lock, if the compiler isn't
// aware that the mutex is in fact locked at this point.)
struct MovableMutexLockTest {
  mutex_lock GetLock() { return mutex_lock{mu}; }
  void Test() { mutex_lock lock = GetLock(); }
  mutex mu;
};
struct SharedMutexLockTest {
  tf_shared_lock GetLock() { return tf_shared_lock{mu}; }
  void Test() { tf_shared_lock lock = GetLock(); }
  mutex mu;
};

}  // namespace
}  // namespace tensorflow
