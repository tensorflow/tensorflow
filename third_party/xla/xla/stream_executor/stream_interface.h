/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_STREAM_INTERFACE_H_
#define XLA_STREAM_EXECUTOR_STREAM_INTERFACE_H_

#include <variant>

#include "xla/stream_executor/platform.h"

namespace stream_executor {
// Pointer-to-implementation object type (i.e. the Stream class delegates to
// this interface) with virtual destruction. This class exists for the
// platform-dependent code to hang any kernel data/resource info/functionality
// off of.
class StreamInterface {
 public:
  // Default constructor for the abstract interface.
  StreamInterface() = default;

  // Default destructor for the abstract interface.
  virtual ~StreamInterface() = default;

  // Sets priority for a stream.
  virtual void SetPriority(StreamPriority priority) {}
  virtual void SetPriority(int priority) {}

  // Gets priority for a stream.
  virtual std::variant<StreamPriority, int> priority() const {
    return StreamPriority::Default;
  }

  // Returns a pointer to a platform specific stream associated with this object
  // if it exists, or nullptr otherwise. This is available via Stream public API
  // as Stream::PlatformSpecificHandle, and should not be accessed directly
  // outside of a StreamExecutor package.
  virtual void* platform_specific_stream() { return nullptr; }

 private:
  StreamInterface(const StreamInterface&) = delete;
  void operator=(const StreamInterface&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_INTERFACE_H_
