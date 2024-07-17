/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_
#define TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "tsl/platform/types.h"

namespace tsl {
namespace profiler {

// Backend for ScopedAnnotation.
class AnnotationStack {
 public:
  using Generator = std::function<std::string_view()>;

  // One level of annotation name will be append to annotations for the current
  // thread, separated by "::".
  // The choice of separator "::" is based on characters not used by TensorFlow
  // for its TensorOps.
  // For complex annotation, use generator so that when not enabled, the
  // generator is not called. The generator is only called when tracing is
  // enabled. Generator and its internal state must be valid before it is
  // popped.
  static void PushAnnotationGenerator(Generator generator);

  static void PushAnnotation(std::string_view name) {
    PushAnnotationGenerator([=] { return name; });
  }

  // For generator that returns std::string, the string will be saved in a
  // wrapper functor, and string_view of it will be returned.
  template <
      typename NameGeneratorT,
      std::enable_if_t<std::is_invocable_v<NameGeneratorT>, bool> = true,
      std::enable_if_t<
          std::is_same_v<std::invoke_result_t<NameGeneratorT>, std::string>,
          bool> = true>
  static void PushAnnotation(NameGeneratorT&& gen) {
    std::string str;
    PushAnnotationGenerator(
        [str, gen = std::forward<NameGeneratorT>(gen)]() mutable {
          str = gen();
          return std::string_view(str);
        });
  }

  // For generator that returns std::string_view, std::string&, const
  // std::string&, const char*, direct convert the return value to
  // std::string_view.
  template <
      typename NameGeneratorT,
      std::enable_if_t<std::is_invocable_v<NameGeneratorT>, bool> = true,
      std::enable_if_t<
          !std::is_same_v<std::invoke_result_t<NameGeneratorT>, std::string>,
          bool> = true>
  static void PushAnnotation(NameGeneratorT&& gen) {
    PushAnnotationGenerator([gen = std::forward<NameGeneratorT>(gen)]() {
      return static_cast<std::string_view>(gen());
    });
  }

  // Resizes the annotation stack for the current thread.
  static void PopAnnotation();

  // Returns the annotation stack for the current thread.
  static const string& Get();

  // Enables or disables the annotation stack.
  static void Enable(bool enable);

  // Returns whether the annotation stack is enabled.
  static bool IsEnabled() {
    return generation_.load(std::memory_order_acquire) & 1;
  }

 private:
  AnnotationStack() = default;

  // Enabled if odd, disabled if even. The value is incremented for every call
  // to Enable() which changes the enabled state.
  static std::atomic<int> generation_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_
