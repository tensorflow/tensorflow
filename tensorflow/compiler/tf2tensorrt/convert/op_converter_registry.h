/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_REGISTRY_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_REGISTRY_H_

#include <initializer_list>
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <array>
#include <type_traits>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

class OpConverterRegistry {
 public:
  OpConverterRegistry();
  ~OpConverterRegistry() = default;

  InitOnStartupMarker Register(const string& name, const int priority,
                               OpConverter converter);

  InitOnStartupMarker Register(const std::initializer_list<std::string>& names,
                               const int priority, OpConverter converter) {
    for (const auto& name : names) {
      Register(name, priority, converter);
    }
    return {};
  }

  template <typename T,
            typename std::enable_if<std::is_convertible<
                typename T::value_type, std::string>::value>::type* = nullptr>
  InitOnStartupMarker Register(const T& names, const int priority,
                               OpConverter converter) {
    for (const auto& name : names) {
      Register(name, priority, converter);
    }
    return {};
  }

  // Clear all registered converters for the given Tensorflow operation name.
  void Clear(const std::string& name);

  StatusOr<OpConverter> LookUp(const string& name);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

OpConverterRegistry* GetOpConverterRegistry();

class RegisterOpConverter {
 public:
  RegisterOpConverter(const string& name, const int priority,
                      OpConverter converter) {
    GetOpConverterRegistry()->Register(name, priority, converter);
  }
};

constexpr int kDefaultConverterPriority = 1;

}  // namespace convert
}  // namespace tensorrt

#define REGISTER_TRT_OP_CONVERTER_IMPL(ctr, func, priority, ...)    \
  static ::tensorflow::InitOnStartupMarker const                    \
      register_trt_op_converter##ctr TF_ATTRIBUTE_UNUSED =          \
          TF_INIT_ON_STARTUP_IF(true)                               \
          << tensorrt::convert::GetOpConverterRegistry()->Register( \
                 __VA_ARGS__, priority, func)

#define REGISTER_TRT_OP_CONVERTER(func, priority, ...)               \
  TF_NEW_ID_FOR_INIT(REGISTER_TRT_OP_CONVERTER_IMPL, func, priority, \
                     __VA_ARGS__)

#define REGISTER_DEFAULT_TRT_OP_CONVERTER(func, ...) \
  REGISTER_TRT_OP_CONVERTER(                         \
      func, tensorrt::convert::kDefaultConverterPriority, __VA_ARGS__)

}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_REGISTRY_H_
