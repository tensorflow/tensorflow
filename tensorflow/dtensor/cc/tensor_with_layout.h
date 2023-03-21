/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_CC_TENSOR_WITH_LAYOUT_H_
#define TENSORFLOW_DTENSOR_CC_TENSOR_WITH_LAYOUT_H_

#include <optional>
#include <string>
#include <vector>

#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {

enum TensorType {
  kDense = 0,
  kResource = 1,
  kSparse = 2,
};

struct EmbeddingResourceAttrs {
  int64_t table_id;
  std::optional<int64_t> slot_id;  // NOLINT
  bool is_dirty = false;
};

class ConstValueNode {
 public:
  explicit ConstValueNode(std::optional<NodeDef> const_value)
      : const_value_(const_value),
        input_layout_for_shape_op_result_(std::nullopt) {}

  // Small constant value optimization for non-resource-handle tensors.
  void set_const_value(NodeDef& const_node) {
    // If we extracted a constant value from the tensor, check if this
    // value was the output from `tf.shape`. In this case, we need to
    // forward the kShapeOpInputLayout attribute to the new node def. This
    // is needed for layout propagation when running in op-by-op mode.
    //
    // TODO(b/162747667): Improve the presentation for Shape input Op
    //                    layout.
    if (input_layout_for_shape_op_result_.has_value()) {
      AddNodeAttr(kShapeOpInputLayout,
                  {input_layout_for_shape_op_result_->ToString()},
                  &(const_node));
    }
    const_value_.emplace(const_node);
  }

  // Clears the cached const value if present.
  void reset_const_value() { const_value_.reset(); }

  const std::optional<NodeDef>& const_value() const { return const_value_; }

  void set_input_layout_for_shape_op_result(const Layout& layout) {
    input_layout_for_shape_op_result_.emplace(layout);
  }

  const std::optional<Layout>& shape_metadata_layout() const {
    return input_layout_for_shape_op_result_;
  }

 private:
  // The value of a small, non-resource tensor. Small constants
  // are directly folded into the SPMD graph instead of being passed as inputs.
  // This provides extra information to the layout propagation and SPMD passes
  // during op-by-op execution. (For example, the reduction indices for Sum,
  // target shapes for Rng/Reshape, etc).
  std::optional<NodeDef> const_value_;

  // The original input layout for a shape Op returned Tensor.
  // This is used to preserve information for a shape op output so that future
  // uses could recover local shape.
  std::optional<Layout> input_layout_for_shape_op_result_ = std::nullopt;
};

// The representation of tensors transferred to underlying devices and the
// layout for the tensors.
class TensorWithLayout
    : public llvm::RTTIExtends<TensorWithLayout, llvm::RTTIRoot> {
 public:
  // Gets the layout for the tensors.
  virtual const Layout& layout() const = 0;

  // Gets the tensor type which indicates whether the tensors are dense,
  // resource or sparse.
  virtual TensorType tensor_type() const = 0;

  // Gets the data type of tensors.
  virtual TF_DataType dtype() const = 0;

  // Encodes the NodeDef via provided builder, if applicable.
  virtual void EncodeAttributes(tensorflow::NodeDefBuilder& builder) const = 0;

  // Generates a key which can be used for SPMD lowering.
  virtual tensorflow::Fprint128 CacheKey() const = 0;

  // Gets the tensor handle at position `index`. This makes sense only when the
  // implementation owns a list of tensor handles. Otherwise this returns
  // `nullptr`.
  virtual TFE_TensorHandle* get_tensor(size_t index) const = 0;

  // Gets the number of tensors.
  virtual size_t num_tensors() const = 0;

  // Returns a string which includes just the value and layout of the tensors.
  virtual std::string SummarizeValue() const = 0;

  // Returns a string which includes `SummarizeValue` along with shape and type
  // information.
  virtual std::string DebugString() const = 0;

  // Gets the mesh for the tensors.
  virtual const Mesh& mesh() const = 0;

  // Computes global shape from layout & local tensor shape.
  //
  // For replicated layout tensors, global shape is simply the shape of local
  // tensors on each device. For sharded tensor, this is the global shape
  // encodes layout & local shape on each device.
  virtual std::vector<int64_t> global_shape() const = 0;

  // Gets a `ConstValueNode` which can operate on a `NodeDef` representing a
  // small const tensor. If it is not null, it can be used in the SPMD
  // expansion, regardless of which runtime is being used.
  virtual ConstValueNode* const_value_node() const = 0;

  // llvm::RTTIExtends ID.
  static char ID;  // NOLINT
};

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_TENSOR_WITH_LAYOUT_H_
