/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PLATFORM_SUBPROCESS_H_
#define XLA_TSL_PLATFORM_SUBPROCESS_H_

#include <memory>
#include <vector>

#include "tsl/platform/types.h"

namespace tsl {

// Channel identifiers.
enum Channel {
  CHAN_STDIN = 0,
  CHAN_STDOUT = 1,
  CHAN_STDERR = 2,
};

// Specify how a channel is handled.
enum ChannelAction {
  // Close the file descriptor when the process starts.
  // This is the default behavior.
  ACTION_CLOSE,

  // Make a pipe to the channel.  It is used in the Communicate() method to
  // transfer data between the parent and child processes.
  ACTION_PIPE,

  // Duplicate the parent's file descriptor. Useful if stdout/stderr should
  // go to the same place that the parent writes it.
  ACTION_DUPPARENT,
};

// Supports spawning and killing child processes.
class SubProcess;

// Returns an object that represents a child process that will be
// launched with the given command-line arguments `argv`. The process
// must be explicitly started by calling the Start() method on the
// returned object.
std::unique_ptr<SubProcess> CreateSubProcess(const std::vector<string>& argv);

}  // namespace tsl

#include "tsl/platform/platform.h"

#if defined(PLATFORM_GOOGLE)
#include "xla/tsl/platform/google/subprocess.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||    \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_POSIX_IOS) || \
    defined(PLATFORM_GOOGLE_IOS)
#include "xla/tsl/platform/default/subprocess.h"  // IWYU pragma: export
#elif defined(PLATFORM_WINDOWS)
#include "xla/tsl/platform/windows/subprocess.h"  // IWYU pragma: export
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

#endif  // XLA_TSL_PLATFORM_SUBPROCESS_H_
