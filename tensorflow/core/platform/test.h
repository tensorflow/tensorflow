/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_PLATFORM_TEST_H_
#define TENSORFLOW_PLATFORM_TEST_H_

#include <memory>
#include <vector>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_GOOGLE_ANDROID)
#include "tensorflow/core/platform/google/build_config/gunit.h"
#else
#include <gtest/gtest.h>
#endif

namespace tensorflow {
namespace testing {

// Return a temporary directory suitable for temporary testing files.
string TmpDir();

// Returns the path to TensorFlow in the directory containing data
// dependencies.
string TensorFlowSrcRoot();

// Return a random number generator seed to use in randomized tests.
// Returns the same value for the lifetime of the process.
int RandomSeed();

// Supports spawning and killing child processes, for use in
// multi-process testing.
class SubProcess {
 public:
  virtual ~SubProcess() {}

  // Starts the subprocess. Returns true on success, otherwise false.
  // NOTE: This method is not thread-safe.
  virtual bool Start() = 0;

  // Kills the subprocess with the given signal number. Returns true
  // on success, otherwise false.
  // NOTE: This method is not thread-safe.
  virtual bool Kill(int signal) = 0;

 protected:
  SubProcess() {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SubProcess);
};

// Returns an object that represents a child process that will be
// launched with the given command-line arguments `argv`. The process
// must be explicitly started by calling the Start() method on the
// returned object.
std::unique_ptr<SubProcess> CreateSubProcess(const std::vector<string>& argv);

// Returns an unused port number, for use in multi-process testing.
// NOTE: This function is not thread-safe.
int PickUnusedPortOrDie();

}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_TEST_H_
