/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_RESOURCE_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_RESOURCE_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace tensorflow {

// Represents a resource, such as a Variable or TensorArray.
class XlaResource {
 public:
  enum Kind {
    kInvalid,
    kVariable,
    kTensorArray,
    kStack,
  };
  static absl::string_view KindToString(Kind kind);

  // Creates a new Stack resource.
  static std::unique_ptr<XlaResource> CreateStack(string name, DataType type,
                                                  int64_t max_size);

  // Creates a new TensorArray resource.
  static std::unique_ptr<XlaResource> CreateTensorArray(
      string name, DataType type, TensorShape shape, xla::XlaOp initial_value,
      int64_t max_array_size);

  XlaResource(Kind kind, int arg_num, string name, DataType type,
              TensorShape shape, xla::XlaOp initial_value,
              int64_t max_array_size,
              const std::set<string>& tensor_array_gradients,
              bool tensor_array_multiple_writes_aggregate,
              const std::optional<ManagedStackTrace>& definition_stack_trace =
                  std::nullopt);

  XlaResource(const XlaResource&) = delete;
  XlaResource(XlaResource&&) = delete;
  XlaResource& operator=(const XlaResource&) = delete;
  XlaResource& operator=(XlaResource&&) = delete;

  Kind kind() const { return kind_; }

  // If this resource is visible externally to the computation, what was its
  // argument number?
  // < 0 means "not visible externally".
  int arg_num() const { return arg_num_; }

  // A descriptive name for the resource, used in error messages.
  const string& name() const { return name_; }

  // Current type and value of the resource. Uninitialized resources are
  // represented by a default (zero) handle and type DT_INVALID.
  // While the type of a resource is notionally fixed during execution, when
  // a resource is first initialized we do not yet know its type, so we keep
  // track of its type dynamically.
  DataType type() const { return type_; }

  // Shape of the resource. For an uninitialized resource, this is ignored.
  // For a Variable, this is the shape of the value. For a TensorArray or Stack
  // this is the shape of each entry in the TensorArray/Stack.
  const TensorShape& shape() const { return shape_; }

  const xla::XlaOp& value() const { return value_; }

  // Value of the resource at computation entry. Used to detect which
  // variables have new values that need to be written back.
  const xla::XlaOp& initial_value() const { return initial_value_; }

  // An xla shape that indicates how this resource variable is represented on
  // device.
  const std::optional<xla::Shape>& representation_shape() const {
    return representation_shape_;
  }

  // A variable is initialized if it has a value.
  bool initialized() const { return value_.valid(); }

  // Sets the type and shape of the resource. The type and shape of a resource
  // must not change once the variable has been initialized.
  Status SetTypeAndShape(DataType type, const TensorShape& shape);

  // Sets the current value of the resource. Returns an error if the type is not
  // set to a valid value.
  Status SetValue(const xla::XlaOp& value);

  // Sets the current value of the resource to an all-zero value.
  Status SetZeroValue(xla::XlaBuilder* builder);

  // Sets the representational shape of the resource on device.
  void SetRepresentationShape(const xla::Shape& shape) {
    representation_shape_ = absl::make_optional(shape);
  }

  // Looks up the gradient for `source`, or creates it if it does not already
  // exist. The call target must be an initialized TensorArray resource. A
  // TensorArray can have multiple named gradients; see the operator
  // documentation for TensorArrayGradV3 for details.
  Status GetOrCreateTensorArrayGradient(const string& source,
                                        xla::XlaBuilder* builder,
                                        XlaResource** gradient_out);

  // Packs a resource into a single XLA value `pack`, suitable for use as
  // an XlaCompiler::Argument. For non-TensorArrays or TensorArrays without
  // gradients, sets `*pack` to `value`.
  // For TensorArrays with gradients, packs the value and its gradient values in
  // a tuple; the gradients values are packed in order by source name.
  Status Pack(xla::XlaOp* pack, xla::XlaBuilder* builder) const;

  // Updates the resource with values from `pack`. If `gradient_sources` is
  // non-empty, treats `pack` as a tuple that represents a TensorArray and
  // its gradients, and unpacks and updates the gradient resources.
  // If `reset_initial_values` is true, sets the initial_values as well as the
  // values.
  // Opposite of Pack().
  Status SetFromPack(const std::set<string>& gradient_sources,
                     const xla::XlaOp& pack, xla::XlaBuilder* builder);

  bool IsOverwritten() { return is_overwritten_; }

  // TensorArray and Stack specific fields
  // TODO(phawkins): refactor this code to use subclasses, rather than putting
  // kind-specific fields in XlaResource.

  // 'max_array_size' stores the expected size of the TensorArray or Stack.
  // We need to store this since sometimes TensorArrays must be initialized
  // lazily since we do not know the element shape at construction time.
  // Used by both TensorArrays and Stacks.
  int64_t max_array_size() const { return max_array_size_; }
  void set_max_array_size(int64_t size) { max_array_size_ = size; }

  bool tensor_array_multiple_writes_aggregate() const {
    return tensor_array_multiple_writes_aggregate_;
  }

  // 'tensor_array_gradient' is a map from TensorArrayGradV3 'source' attributes
  // to an XlaResource containing the gradient TensorArrays. We store a pointer
  // here since there should only be one gradient TensorArray per 'source'
  // string, irrespective of the number of calls to TensorArrayGrad. The map
  // is ordered since values are packed into tuples by Pack() sorted by name
  // order.
  const std::map<string, std::unique_ptr<XlaResource>>& tensor_array_gradients()
      const {
    return tensor_array_gradients_;
  }

 private:
  const Kind kind_;
  const int arg_num_;
  const string name_;

  DataType type_;
  TensorShape shape_;
  xla::XlaOp value_;
  xla::XlaOp initial_value_;

  // An xla shape that indicates how this resource variable is represented on
  // device.
  std::optional<xla::Shape> representation_shape_;

  int64_t max_array_size_ = -1;
  bool tensor_array_multiple_writes_aggregate_ = false;

  std::map<string, std::unique_ptr<XlaResource>> tensor_array_gradients_;
  bool is_overwritten_ = false;

  std::optional<ManagedStackTrace> definition_stack_trace_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_RESOURCE_H_
