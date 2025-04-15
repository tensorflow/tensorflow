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
/// \file
///
/// Functions for serializiation/deserialization of control dependency
/// information to/from model metadata.
///

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_METADATA_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_METADATA_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/compiler/mlir/lite/utils/control_edges.h"

namespace tflite {

/// Control dependencies for the model is the collection of control dependencies
/// for its subgraphs.
using ModelControlDependencies = std::vector<ControlEdges>;

/// Serializes `in` into the returned string. The result is parseable with
/// ParseModelControlDependencies.
std::string SerializeModelControlDependencies(
    const ModelControlDependencies& in);

/// Deserializes `*out` from a character buffer of size `size` at `data`.
/// Returns true iff successful. `*out` needn't be empty before invocation.
/// When returning false, `*out`'s state is undefined.
bool ParseModelControlDependencies(const char* data, size_t size,
                                   ModelControlDependencies* out);

/// The key under which to store the serialized control dependencies in the
/// model's metadata.
constexpr char kModelControlDependenciesMetadataKey[] =
    "model_control_dependencies";

/// To allow future changes to the format, serialized control dependency data
/// will contain a version; this constant is the version that will be used for
/// serialization.  For deserialization, past versions should remain parseable.
constexpr uint32_t kModelControlDependenciesMetadataVersion = 1;

inline constexpr char kModelUseStablehloTensorKey[] = "keep_stablehlo_constant";

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_METADATA_UTIL_H_
