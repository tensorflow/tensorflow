/* Copyright 2018 The OpenXLA Authors.

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

// IWYU pragma: private, include "xla/stream_executor/platform/initialize.h"

#ifndef XLA_STREAM_EXECUTOR_PLATFORM_DEFAULT_INITIALIZE_H_
#define XLA_STREAM_EXECUTOR_PLATFORM_DEFAULT_INITIALIZE_H_

namespace stream_executor {
namespace port {

class Initializer {
 public:
  typedef void (*InitializerFunc)();
  explicit Initializer(InitializerFunc func) { func(); }

  struct Dependency {
    Dependency(const char *n, Initializer *i) : name(n), initializer(i) {}
    const char *const name;
    Initializer *const initializer;
  };

  struct DependencyRegisterer {
    DependencyRegisterer(const char *type, const char *name,
                         Initializer *initializer,
                         const Dependency &dependency);
  };
};

}  // namespace port
}  // namespace stream_executor

#define STREAM_EXECUTOR_REGISTER_INITIALIZER(type, name, body)             \
  static void google_init_##type##_##name() { body; }                      \
  ::stream_executor::port::Initializer google_initializer_##type##_##name( \
      google_init_##type##_##name)

#define STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(name, body) \
  STREAM_EXECUTOR_REGISTER_INITIALIZER(module, name, body)

#define STREAM_EXECUTOR_DECLARE_INITIALIZER(type, name) \
  extern ::stream_executor::port::Initializer google_initializer_##type##_##name

#define STREAM_EXECUTOR_DECLARE_MODULE_INITIALIZER(name) \
  STREAM_EXECUTOR_DECLARE_INITIALIZER(module, name)

#define STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER_SEQUENCE(name1, name2)

#endif  // XLA_STREAM_EXECUTOR_PLATFORM_DEFAULT_INITIALIZE_H_
