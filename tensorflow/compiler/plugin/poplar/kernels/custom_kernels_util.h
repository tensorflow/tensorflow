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

#include "absl/container/flat_hash_set.h"

#include <sstream>
#include <string>

namespace xla {
class HloInstruction;
namespace poplarplugin {
namespace IPUCustomKernelsUtil {
// Returns true if inst is a call to a custom op for Poplibs
const bool IsPoplibsOp(const HloInstruction* inst);

class AttributeMap {
 public:
  AttributeMap();
  AttributeMap(const HloInstruction* custom_call);

  template <typename T>
  void AddAttribute(const std::string& field_name, const T& attr);

  StatusOr<std::string> GetAttributeAsString(
      const std::string& field_name) const;
  StatusOr<float> GetAttributeAsFloat(const std::string& field_name) const;
  StatusOr<int> GetAttributeAsInt(const std::string& field_name) const;
  StatusOr<bool> GetAttributeAsBool(const std::string& field_name) const;
  StatusOr<tensorflow::DataType> GetAttributeAsTFDataType(
      const std::string& field_name) const;
  StatusOr<absl::flat_hash_set<int64>> GetAttributeAsInt64FlatHashSet(
      const std::string& field_name) const;

  const std::string Serialise();

 private:
  Json::Value attributes_;
};
}  // namespace IPUCustomKernelsUtil
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_CUSOM_KERNELS_UTIL_H_
