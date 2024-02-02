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

// Attribute name of a serialized TpuCompileMetadataProto. This is backward
// compatible.
inline constexpr absl::string_view kMetadataAttrName = "tpu_compile_metadata";
// Attribute name of a text TpuCompileMetadataProto. Note that the text proto is
// not backward compatible and only used for debug.
inline constexpr absl::string_view kMetadataTextAttrName =
    "__tpu_compile_metadata_text";

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_CONSTANTS_H_
