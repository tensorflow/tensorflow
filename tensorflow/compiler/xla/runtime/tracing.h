/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_TRACING_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_TRACING_H_

#include <string_view>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"

namespace xla {
namespace runtime {

// XLA run-time representation of the `!rt.hlo_trace` attribute.
struct HloTrace {
  std::string_view hlo_op;
};

// Registers type id names for tracing attributes.
inline void PopulateTraceTypeIdNames(TypeIDNameRegistry& registry) {
  registry.Register<Tagged<HloTrace>>("__type_id_hlo_trace");
}

// Register XLA runtime custom calls attribute decoding.
XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    HloTrace, AggregateMember<std::string_view>("hlo_op"));

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_TRACING_H_
