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

#ifndef TENSORFLOW_STREAM_EXECUTOR_PLATFORM_DEFAULT_INITIALIZE_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLATFORM_DEFAULT_INITIALIZE_H_

#undef REGISTER_MODULE_INITIALIZER
#undef DECLARE_MODULE_INITIALIZER
#undef REGISTER_MODULE_INITIALIZER_SEQUENCE

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

#define REGISTER_INITIALIZER(type, name, body)                             \
  static void google_init_##type##_##name() { body; }                      \
  ::stream_executor::port::Initializer google_initializer_##type##_##name( \
      google_init_##type##_##name)

#define REGISTER_MODULE_INITIALIZER(name, body) \
  REGISTER_INITIALIZER(module, name, body)

#define DECLARE_INITIALIZER(type, name) \
  extern ::stream_executor::port::Initializer google_initializer_##type##_##name

#define DECLARE_MODULE_INITIALIZER(name) DECLARE_INITIALIZER(module, name)

#define REGISTER_MODULE_INITIALIZER_SEQUENCE(name1, name2)

#endif  // TENSORFLOW_STREAM_EXECUTOR_PLATFORM_DEFAULT_INITIALIZE_H_
