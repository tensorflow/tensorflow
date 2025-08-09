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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_CLUSTER_SIGNATURE_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_CLUSTER_SIGNATURE_H_

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Pre-computed device compilation canonical function.
struct DeviceCompilationCanonicalFunction {
  std::string canonical;
};

inline DeviceCompilationCanonicalFunction Canonicalize(
    const NameAttrList& function) {
  return {Canonicalize(function.name(), AttrSlice(&function.attr()))};
}

// Describes the types, shapes and any compile-time constant arguments
// to a kernel. Key that uniquely identifies a compilation output.
struct DeviceCompilationClusterSignature {
  // Name of the cluster, built from the function name and it's attributes.
  std::string name;

  // List of args (either as a TensorTypeAndShape or as a Tensor value)
  // for compile-time constant arguments to the compilation, ordered by
  // argument number. Tensors must be in host memory.
  using TensorTypeAndShape =
      std::pair<DataType, absl::InlinedVector<int64_t, 4>>;
  absl::InlinedVector<std::variant<Tensor, TensorTypeAndShape>, 8> args;

  bool operator==(const DeviceCompilationClusterSignature& other) const;

  struct Hash {
    uint64 operator()(const DeviceCompilationClusterSignature& signature) const;
  };

  // Returns a human-readable description of the signature.
  std::string HumanString() const;

  // Builds the signature for a compilation.
  static absl::StatusOr<DeviceCompilationClusterSignature> Build(
      const NameAttrList& function,
      absl::Span<const XlaCompiler::Argument> args);

  static absl::StatusOr<DeviceCompilationClusterSignature> Build(
      const DeviceCompilationCanonicalFunction& canonical_function,
      absl::Span<const XlaCompiler::Argument> args);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_CLUSTER_SIGNATURE_H_
