/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_HLO_IR_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_
#define XLA_HLO_IR_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla {

class HloModule;

// This class specifies the alias map from output index to parameter number and
// parameter index in the entry computation.
class HloInputOutputAliasConfig {
 public:
  // The kind of aliases which can be set. A kMayAlias is one setup at
  // compilation time by the user, and has to be respected. A kMustAlias one
  // might be setup by the compiler, if it decides it is convenient to do so.
  enum AliasKind {
    kMayAlias,
    kMustAlias,
  };
  // Defines the alias information for a given output buffer. A given output
  // buffer shape index can refer only to one parameter+index.
  struct Alias {
    Alias(int64_t parameter_number, ShapeIndex parameter_index,
          AliasKind kind = kMayAlias)
        : parameter_number(parameter_number),
          parameter_index(std::move(parameter_index)),
          kind(kind) {}

    int64_t parameter_number;
    ShapeIndex parameter_index;
    AliasKind kind;

    bool must_alias() const { return kind == kMustAlias; }

    std::string ToString() const {
      return absl::StrFormat("(%lld, %s, %s)", parameter_number,
                             parameter_index.ToString(),
                             kind == kMustAlias ? "must-alias" : "may-alias");
    }
  };

  HloInputOutputAliasConfig() = default;

  explicit HloInputOutputAliasConfig(Shape output_shape)
      : alias_(std::move(output_shape)) {}

  virtual ~HloInputOutputAliasConfig() = default;

  // Sets up alias config from `output_index` to `param_index` at
  // `param_number`.
  Status SetUpAlias(const ShapeIndex& output_index, int64_t param_number,
                    const ShapeIndex& param_index,
                    AliasKind must_alias = kMayAlias);

  // Returns true if the given parameter is aliased with one of the output
  // buffers.
  bool ParameterHasAlias(int64_t param_number,
                         const ShapeIndex& param_index) const {
    return GetAliasedOutput(param_number, param_index).has_value();
  }

  // Checks whether the provided output index has already been aliased.
  bool OutputHasAlias(const ShapeIndex& output_index) const;

  // (De)Serializes an HloInputOutputAliasConfig to/from an
  // HloInputOutputAliasProto.
  HloInputOutputAliasProto ToProto() const;

  static absl::StatusOr<HloInputOutputAliasConfig> CreateFromProto(
      Shape output_shape, const HloInputOutputAliasProto& proto);

  // Returns the output index that the given parameter and parameter index is
  // aliased with. A nullopt is returned if there is no output that is aliased
  // with the parameter number and index.
  std::optional<ShapeIndex> GetAliasedOutput(
      int64_t param_number, const ShapeIndex& param_index) const;

  // Returns the number of parameter and index of the parameter buffer that the
  // given output buffer index is aliased with. A nullopt is returned if there
  // is no parameter is aliased with the specific output.
  std::optional<Alias> GetAliasedParameter(
      const ShapeIndex& output_index) const;

  // Returns if the parameter at the given parameter number and parameter
  // index must-alias with an output.
  bool ParameterMustAlias(int64_t param_number,
                          const ShapeIndex& param_index) const;

  using AliasFn =
      absl::FunctionRef<void(const ShapeIndex& output_index, const Alias&)>;

  // Iterates through each aliased output and input.
  void ForEachAlias(AliasFn fn) const;

  using AliasFnWithStatus =
      absl::FunctionRef<Status(const ShapeIndex& output_index, const Alias&)>;

  // Verifies that the given config is valid for the given module.
  // Specifically, the config's input and output should be in-bound and size of
  // the aliased buffers should match.
  Status Verify(const HloModule& module,
                absl::FunctionRef<int64_t(const Shape&)> size_func) const;

  Status ForEachAliasWithStatus(AliasFnWithStatus fn) const;

  // Returns the shape of the output of the alias config.
  const Shape& shape() const;

  std::string ToString() const;

  std::string ToShortString() const;

 private:
  // A ShapeTree which indicates the list of buffers that's expected to be
  // aliased. The key on this shape tree represents the output index. The value
  // is an Alias data structure which defines the input parameter coordinates.
  // If the value is nullopt, it means there is no parameter aliasing for this
  // output.
  ShapeTree<std::optional<Alias>> alias_;
};

// This class specifies donors of the input buffer (specified by parameter
// number and parameter index in the entry computation). The donated buffer can
// be matched with any valid output tensor, which differs from
// HloInputOutputAliasConfig.
class HloBufferDonorConfig {
 public:
  // Defines a input buffer donor. In real world, organ donors refer to the
  // persons agreeing to remove their organs (usually after death). Similarly, a
  // registered buffer donor indicates that the input parameter can be removed
  // when there is no dependency. Therefore, the memory buffer can be reused by
  // a matched output.
  struct BufferDonor {
    BufferDonor(int64_t param_number, ShapeIndex param_index)
        : param_number(param_number), param_index(std::move(param_index)) {}

    int64_t param_number;
    ShapeIndex param_index;

    bool operator==(const BufferDonor& other) const {
      return param_number == other.param_number &&
             param_index == other.param_index;
    }

    bool operator<(const BufferDonor& other) const {
      return std::forward_as_tuple(param_number, param_index) <
             std::forward_as_tuple(other.param_number, other.param_index);
    }
    bool operator>(const BufferDonor& other) const { return other < *this; }
    bool operator<=(const BufferDonor& other) const { return !(*this > other); }
    bool operator>=(const BufferDonor& other) const { return !(*this < other); }

    // A hash function borrowed from go/absl-hash.
    template <typename H>
    friend H AbslHashValue(H h, const BufferDonor& donor) {
      return H::combine(std::move(h), donor.param_number, donor.param_index);
    }
  };

  HloBufferDonorConfig() = default;
  virtual ~HloBufferDonorConfig() = default;

  // Register and unregister the parameter with `param_index` at `param_number`
  // as a buffer donor.
  Status AddBufferDonor(int64_t param_number, const ShapeIndex& param_index);
  Status RemoveBufferDonor(int64_t param_number, const ShapeIndex& param_index);

  // Returns true if the given parameter is registered as a buffer donor.
  bool ParameterIsBufferDonor(int64_t param_number,
                              const ShapeIndex& param_index) const;

  // (De)Serializes an HloBufferDonorConfig to/from an HloBufferDonorProto.
  HloBufferDonorProto ToProto() const;
  static absl::StatusOr<HloBufferDonorConfig> CreateFromProto(
      const HloBufferDonorProto& proto);

  // Verifies that the given config is valid for the given module.
  // The config's input should be in-bound and this config cannot overlap with
  // the given module's input_output_alias_config.
  Status Verify(const HloModule& module) const;

  // Returns the registered buffer donors
  const absl::btree_set<BufferDonor>& buffer_donor() const {
    return buffer_donor_;
  }

  std::string ToString() const;

  std::string ToShortString() const;

 private:
  // A set recording the registered buffer donors.
  absl::btree_set<BufferDonor> buffer_donor_;
};

std::ostream& operator<<(std::ostream& out,
                         const HloInputOutputAliasConfig& config);
std::ostream& operator<<(std::ostream& out, const HloBufferDonorConfig& config);

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_
