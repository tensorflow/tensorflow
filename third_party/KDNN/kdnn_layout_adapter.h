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

#ifndef TENSORFLOW_CORE_UTIL_KDNN_LAYOUT_ADAPTER_H_
#define TENSORFLOW_CORE_UTIL_KDNN_LAYOUT_ADAPTER_H_
#include "kdnn.hpp"

namespace KDNN {

template <int Rank, bool Transposed = false>
struct LayoutAdapter {
    static constexpr Layout value = Layout::UNDEFINED;
};

template <>
struct LayoutAdapter<1, false> {
    static constexpr Layout value = Layout::A;
};

template <>
struct LayoutAdapter<2, false> {
    static constexpr Layout value = Layout::AB;
};

template <>
struct LayoutAdapter<2, true> {
    static constexpr Layout value = Layout::BA;
};
} // KDNN
#endif  // TENSORFLOW_CORE_UTIL_KDNN_LAYOUT_ADAPTER_H