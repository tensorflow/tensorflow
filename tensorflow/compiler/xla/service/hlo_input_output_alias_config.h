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
  HloInputOutputAliasConfig() = default;

  explicit HloInputOutputAliasConfig(Shape shape) : alias_(shape) {}

  virtual ~HloInputOutputAliasConfig() = default;

  // Sets up alias config from `output_index` to `param_index` at
  // `param_number`.
  Status SetUpAlias(const ShapeIndex& output_index, int64 param_number,
                    const ShapeIndex& param_index);

  // Returns true if the given parameter is aliased with one of the output
  // buffers.
  bool ParameterHasAlias(int64 param_number,
                         const ShapeIndex& param_index) const;

  // Checks whether the provided output index has already been aliased.
  bool OutputHasAlias(const ShapeIndex& output_index) const;

  // (De)Serializes an HloInputOutoutAliasConfig to/from an
  // HloInputOutoutAliasProto.
  HloInputOutputAliasProto ToProto() const;

  static StatusOr<HloInputOutputAliasConfig> CreateFromProto(
      const Shape& output_shape, const HloInputOutputAliasProto& proto);

  // Returns the output index that the given parameter and parameter index is
  // aliased with. A nullopt is returned if there is no output that is aliased
  // with the parameter number and index.
  absl::optional<ShapeIndex> GetAliasedOutput(
      int64 param_number, const ShapeIndex& param_index) const;

  // Returns the number of parameter and index of the parameter buffer that the
  // given output buffer index is aliased with. A nullopt is returned if there
  // is no parameter is aliased with the specific output.
  absl::optional<std::pair<int64, ShapeIndex>> GetAliasedParameter(
      const ShapeIndex& output_index) const;

  using AliasFn =
      std::function<void(const ShapeIndex& output_index, int64 param_number,
                         const ShapeIndex& param_index)>;

  // Iterates through each aliased output and input.
  void ForEachAlias(AliasFn fn) const;

  using AliasFnWithStatus =
      std::function<Status(const ShapeIndex& output_index, int64 param_number,
                           const ShapeIndex& param_index)>;

  // Verifies that the given config is valid for the given module.
  // Specifically, the config's input and output should be in-bound and size of
  // the aliased buffers should match.
  Status Verify(const HloModule& module,
                std::function<int64(const Shape&)> size_func_) const;

  Status ForEachAliasWithStatus(AliasFnWithStatus fn) const;

  string ToString() const;

 private:
  // A ShapeTree which indicates the list of buffers that's expected to be
  // aliased. The key on this shape tree represents the output index. The value
  // is a pair of parameter number and index into the buffer. If the value is
  // nullopt, it means there is no parameter aliasing for this output.
  ShapeTree<absl::optional<std::pair<int64, ShapeIndex>>> alias_;

  // The indices of the output which have been aliased.
  absl::flat_hash_set<ShapeIndex> aliased_output_indices_;
};

std::ostream& operator<<(std::ostream& out,
                         const HloInputOutputAliasConfig& config);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_
