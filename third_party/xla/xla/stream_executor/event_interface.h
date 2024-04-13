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

#ifndef XLA_STREAM_EXECUTOR_EVENT_INTERFACE_H_
#define XLA_STREAM_EXECUTOR_EVENT_INTERFACE_H_

namespace stream_executor {

// Base class for all kinds of Events supported by StreamExecutors.
class EventInterface {
 public:
  EventInterface() = default;
  virtual ~EventInterface() = default;

 private:
  EventInterface(const EventInterface&) = delete;
  void operator=(const EventInterface&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_EVENT_INTERFACE_H_
