/* Copyright 2018 Graphcore Ltd

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_CUSOM_KERNELS_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_CUSOM_KERNELS_UTIL_H_

#include "tensorflow/compiler/xla/status_macros.h"

#include "include/json/json.h"
#include "tensorflow/core/framework/types.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/any.h"

#include <sstream>
#include <string>

namespace xla {
class HloInstruction;

namespace poplarplugin {

enum class PoplibsLib : uint32 {
  Poplin = 0,
  Popnn,
  Popops,
  Poprand,
  // Do not add beyond this point.
  _NumLibs
};
std::string PoplibsLibToString(const PoplibsLib&);
absl::optional<PoplibsLib> StringToPoplibsLib(const std::string&);

enum class PoplibsOp : uint32 {
  // Poplin:
  // Popnn:
  LstmLayerFwd = 0,
  LstmLayerBwd,
  GroupNormInference,
  GroupNormTraining,
  GroupNormGrad,
  GroupNormStatistics,
  // Popops:
  Sqrt,
  Rsqrt,
  // Poprand:
  // Do not add beyond this point.
  _NumOps
};
std::string PoplibsOpToString(const PoplibsOp&);
absl::optional<PoplibsOp> StringToPoplibsOp(const std::string&);

// Function used to get the string target for the kCustomCall
std::string GetPoplibsCustomOpTargetString(const PoplibsLib&, const PoplibsOp&);
// Tried to convert the string target for the kCustomCall
absl::optional<std::pair<PoplibsLib, PoplibsOp>> GetPoplibsCustomOp(
    const HloInstruction* inst);

// Returns true if inst is a call to a custom op for Poplibs
const bool IsPoplibsCustomOp(const HloInstruction* inst);
// Returns true if inst is a call to a custom op for Poplibs of a certain type.
const bool IsPoplibsCustomOp(const HloInstruction* inst,
                             const PoplibsLib& poplibs_lib,
                             const PoplibsOp& poplibs_op);
// Returns true if inst is a call to a custom op for Poplibs which is
// elementwise.
const bool IsPoplibsCustomOpElementwise(const HloInstruction* inst);

namespace IPUCustomKernelsUtil {

class AttributeMap {
 public:
  AttributeMap();
  AttributeMap(const HloInstruction* custom_call);

  // We support:
  // * float, int, bool, uint64, int64, tensorflow::DataType
  // * absl::flat_hash_set of int64
  // * absl::flat_hash_map of int64 to int64
  void AddAttribute(const std::string& field_name, const absl::any& attr);

  StatusOr<std::string> GetAttributeAsString(
      const std::string& field_name) const;
  StatusOr<float> GetAttributeAsFloat(const std::string& field_name) const;
  StatusOr<int> GetAttributeAsInt(const std::string& field_name) const;
  StatusOr<uint64> GetAttributeAsUInt64(const std::string& field_name) const;
  StatusOr<bool> GetAttributeAsBool(const std::string& field_name) const;
  StatusOr<tensorflow::DataType> GetAttributeAsTFDataType(
      const std::string& field_name) const;
  StatusOr<absl::flat_hash_set<int64>> GetAttributeFlatHashSet(
      const std::string& field_name) const;
  StatusOr<absl::flat_hash_map<int64, int64>> GetAttributeFlatHashMap(
      const std::string& field_name) const;

  const std::string Serialise();

 private:
  Json::Value attributes_;
};
}  // namespace IPUCustomKernelsUtil
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_CUSOM_KERNELS_UTIL_H_
