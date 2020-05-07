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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

class HloModule;

// This class specifies the alias map from output index to parameter number and
// parameter index in the entry computation.
class HloInputOutputAliasConfig {
 public:
  // The kind of aliases which can be set. A kUserAlias is one setup at
  // compilation time by the user, and has to be respected. A kSystemAlias one
  // might be setup by the compiler, if it decides it is convenient to do so.
  enum AliasKind {
    kNoAlias,
    kUserAlias,
    kSystemAlias,
  };

  // Defines the alias information for a given output buffer. A given output
  // buffer shape index can refer only to one parameter+index.
  struct Alias {
    Alias(AliasKind kind, int64 parameter_number, ShapeIndex parameter_index)
        : kind(kind),
          parameter_number(parameter_number),
          parameter_index(std::move(parameter_index)) {}

    AliasKind kind;
    int64 parameter_number;
    ShapeIndex parameter_index;
  };

  HloInputOutputAliasConfig() = default;

  explicit HloInputOutputAliasConfig(Shape output_shape)
      : alias_(std::move(output_shape)) {}

  virtual ~HloInputOutputAliasConfig() = default;

  // Sets up alias config from `output_index` to `param_index` at
  // `param_number`.
  Status SetUpAlias(const ShapeIndex& output_index, int64 param_number,
                    const ShapeIndex& param_index,
                    AliasKind kind = AliasKind::kUserAlias);

  // Returns the kind of alias for the given parameter number and parameter
  // index. If no alias exists, AliasKind::kNoAlias is returned.
  AliasKind ParameterAliasKind(int64 param_number,
                               const ShapeIndex& param_index) const;

  // Returns true if the given parameter is aliased with one of the output
  // buffers.
  bool ParameterHasAlias(int64 param_number,
                         const ShapeIndex& param_index) const {
    return ParameterAliasKind(param_number, param_index) != AliasKind::kNoAlias;
  }

  // Checks whether the provided output index has already been aliased.
  bool OutputHasAlias(const ShapeIndex& output_index) const;

  // (De)Serializes an HloInputOutputAliasConfig to/from an
  // HloInputOutputAliasProto.
  HloInputOutputAliasProto ToProto() const;

  static StatusOr<HloInputOutputAliasConfig> CreateFromProto(
      Shape output_shape, const HloInputOutputAliasProto& proto);

  // Returns the output index that the given parameter and parameter index is
  // aliased with. A nullopt is returned if there is no output that is aliased
  // with the parameter number and index.
  absl::optional<ShapeIndex> GetAliasedOutput(
      int64 param_number, const ShapeIndex& param_index) const;

  // Returns the number of parameter and index of the parameter buffer that the
  // given output buffer index is aliased with. A nullopt is returned if there
  // is no parameter is aliased with the specific output.
  absl::optional<Alias> GetAliasedParameter(
      const ShapeIndex& output_index) const;

  using AliasFn =
      std::function<void(const ShapeIndex& output_index, const Alias&)>;

  // Iterates through each aliased output and input.
  void ForEachAlias(AliasFn fn) const;

  using AliasFnWithStatus =
      std::function<Status(const ShapeIndex& output_index, const Alias&)>;

  // Verifies that the given config is valid for the given module.
  // Specifically, the config's input and output should be in-bound and size of
  // the aliased buffers should match.
  Status Verify(const HloModule& module,
                std::function<int64(const Shape&)> size_func_) const;

  Status ForEachAliasWithStatus(AliasFnWithStatus fn) const;

  // Returns the shape of the output of the alias config.
  const Shape& shape() const;

  string ToString() const;

 private:
  // A ShapeTree which indicates the list of buffers that's expected to be
  // aliased. The key on this shape tree represents the output index. The value
  // is an Alias data structure which defines the input parameter coordinates.
  // If the value is nullopt, it means there is no parameter aliasing for this
  // output.
  ShapeTree<absl::optional<Alias>> alias_;
};

std::ostream& operator<<(std::ostream& out,
                         const HloInputOutputAliasConfig& config);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_
