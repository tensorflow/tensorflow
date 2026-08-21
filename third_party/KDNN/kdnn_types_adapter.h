/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_KDNN_TYPES_ADAPTER_H_
#define TENSORFLOW_CORE_UTIL_KDNN_TYPES_ADAPTER_H_
#include "kdnn.hpp"

namespace KDNN {
namespace Element {

template <typename T>
struct TypeAdapter {
    static constexpr TypeT value = TypeT::UNDEFINED;
};

template <>
struct TypeAdapter<float> {
    static constexpr TypeT value = TypeT::F32;
};

template <>
struct TypeAdapter<Eigen::half> {
    static constexpr TypeT value = TypeT::F16;
};

template <>
struct TypeAdapter<tensorflow::bfloat16> {
    static constexpr TypeT value = TypeT::BF16;
};

template <>
struct TypeAdapter<int32_t> {
    static constexpr TypeT value = TypeT::S32;
};

template <>
struct TypeAdapter<int8_t> {
    static constexpr TypeT value = TypeT::S8;
};

template <>
struct TypeAdapter<uint8_t> {
    static constexpr TypeT value = TypeT::U8;
};
} // Element
} // KDNN

#endif  // TENSORFLOW_CORE_UTIL_KDNN_TYPES_ADAPTER_H_