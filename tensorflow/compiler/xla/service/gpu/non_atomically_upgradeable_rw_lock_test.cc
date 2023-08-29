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

#include "tensorflow/compiler/xla/service/gpu/non_atomically_upgradeable_rw_lock.h"

#include <gtest/gtest.h>
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

TEST(NonAtomicallyUpgradeableRWLock, UpgradeReaderMutexLock) {
  absl::Mutex mu;
  {
    NonAtomicallyUpgradeableRWLock reader_lock(&mu);
    mu.AssertReaderHeld();

    {
      NonAtomicallyUpgradeableRWLock::WriterLock writer_lock =
          reader_lock.UpgradeToWriterMutexLock();
      mu.AssertHeld();
    }

    // The lock downgrades after the WriterLock goes out of scope.
    mu.AssertReaderHeld();
  }
  mu.AssertNotHeld();
}

}  // namespace
}  // namespace gpu
}  // namespace xla
