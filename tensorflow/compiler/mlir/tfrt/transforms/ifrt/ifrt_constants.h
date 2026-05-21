/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_CONSTANTS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_CONSTANTS_H_

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace ifrt_serving {

// Attribute name of a text TpuCompileMetadataProto. Note that the text proto is
// not backward compatible and shall not be serialized.
inline constexpr absl::string_view kMetadataTextAttrName =
    "__tpu_compile_metadata_text";

// Name of a variable as loaded IFRT array .
inline constexpr absl::string_view kVariableArrayNameAttr =
    "__variable_array_name";

// Attribute of a text `VariableDeviceShardingConfigProto`.
inline constexpr absl::string_view kVariableShardingConfigTextAttr =
    "__variable_sharding_config_text";

// Parallel-to-operands i64 array on tf.IfrtCall / tf.AsyncIfrtCallOp (and the
// MLRT-lowered tf_mlrt.ifrt_call / tf_mlrt.async_ifrt_call). Values:
//   -1 : transfer this operand individually (default; the un-packed path).
//   >= 0 : pack into the named transfer group at runtime.
// Set by IfrtPackInputsPlannerPass; consumed by the H2D transfer code in
// IfrtServingExecutable.
inline constexpr absl::string_view kIfrtPackGroupIdsAttr =
    "ifrt_pack_group_ids";

// Parallel-to-operands i64 array carrying the byte offset of each operand
// inside its pack group's host scratch buffer. Only meaningful where the
// corresponding kIfrtPackGroupIdsAttr value is >= 0. Set inside
// IfrtServingExecutable after PackInputsPass computes the SliceInfo layout.
inline constexpr absl::string_view kIfrtPackOffsetsAttr = "ifrt_pack_offsets";

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_CONSTANTS_H_
