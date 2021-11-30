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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_PARAMETER_BINDING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_PARAMETER_BINDING_H_

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

class HloModule;
// We currently use an explicit API that takes an extra parameter to indicate
// the runtime size of a dynamic dimension. DynamicParameterBinding indicates
// the relationship between parameter: We can have a dynamic parameter that
// points to another target parameter to indicate that the target parameter is
// dynamic.
//
//
// TODO(b/119520625): Remove this API once we have more dynamic shape infra
// ready.
class DynamicParameterBinding {
 public:
  // DynamicParameter represents a special parameter that is used to represent
  // the runtime size of a dimension of another parameter. A dynamic parameter
  // has to be a scalar value.
  struct DynamicParameter {
    // The parameter number of dynamic parameter.
    int64_t parameter_num;
    // The index of the parameter.
    ShapeIndex parameter_index;
  };

  // DynamicDimension represents a dimension whose size is determined at
  // runtime. A DynamicDimension's runtime size is determined by the binded
  // DynamicParameter using `DynamicParameterBinding::Bind` method.
  struct DynamicDimension {
    // The parameter number of dynamic dimension.
    int64_t parameter_num;
    // The subshape index of the parameter.
    ShapeIndex parameter_index;
    // The dimension number in the subshape.
    int64_t dimension;

    // "friend" keyword are added so these functions can be found by ADL.
    template <typename H>
    friend H AbslHashValue(H h, const DynamicDimension& m) {
      return H::combine(std::move(h), m.parameter_num, m.parameter_index,
                        m.dimension);
    }

    friend bool operator==(const DynamicDimension& lhs,
                           const DynamicDimension& rhs) {
      return lhs.parameter_num == rhs.parameter_num &&
             lhs.parameter_index == rhs.parameter_index &&
             lhs.dimension == rhs.dimension;
    }
  };

  DynamicParameterBinding() = default;

  virtual ~DynamicParameterBinding() = default;

  // Adds binding which indicates that the dimension indicated by
  // `dynamic_dimension` is dynamic, and its runtime size is represented by
  // `dynamic_parameter`.
  Status Bind(const DynamicParameter& dynamic_parameter,
              const DynamicDimension& dynamic_dimension);

  // Returns the parameter and the index representing the runtime size of
  // dimension `dim_num` of parameter `param_num` at `param_index`.
  //
  // Returns nullopt if the binding is not set.
  absl::optional<DynamicParameter> GetBinding(
      const DynamicDimension& dynamic_dimension) const;

  using BindingFn =
      std::function<Status(const DynamicParameter& dynamic_parameter,
                           const DynamicDimension& dynamic_dimension)>;

  // Iterate through each binding.
  Status ForEachBinding(BindingFn fn) const;

  DynamicParameterBindingProto ToProto() const;

  static StatusOr<DynamicParameterBinding> CreateFromProto(
      const DynamicParameterBindingProto& proto);

  string ToString() const;

  // Verifies that the given binding is valid for the given module.
  // Specifically, the binding's parameter and parameter size should be valid.
  Status Verify(const HloModule& module) const;

 private:
  // Keeps track of mappings from DynamicDimension to DynamicParameter. The
  // direction of is chosen so that we can easily query if a dimension is
  // dynamic and which dynamic parameter represents the real size of that
  // dimension.
  absl::flat_hash_map<DynamicDimension, DynamicParameter> bindings_;
};

std::ostream& operator<<(std::ostream& out,
                         const DynamicParameterBinding& binding);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_PARAMETER_BINDING_H_
