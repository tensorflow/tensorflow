/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_FRAMEWORK_SHAPE_INFERENCE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_FRAMEWORK_SHAPE_INFERENCE_H_

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace shape_inference {

struct DimensionOrConstant;
class InferenceContext;

// Dimension values are accessed through InferenceContext.
class Dimension {
 private:
  Dimension();
  Dimension(int64 value);
  ~Dimension() {}

  const int64 value_;

  friend class InferenceContext;
  TF_DISALLOW_COPY_AND_ASSIGN(Dimension);
};

class DimensionHandle {
 public:
  DimensionHandle() {}

 private:
  DimensionHandle(const Dimension* dim) { ptr_ = dim; }

  const Dimension* operator->() { return ptr_; }
  bool IsSet() const { return ptr_ != nullptr; }
  bool SameHandle(DimensionHandle d) const { return ptr_ == d.ptr_; }

  const Dimension* ptr_ = nullptr;

  friend struct DimensionOrConstant;
  friend class InferenceContext;
  friend class ShapeInferenceTest;
  friend class ShapeInferenceTestutil;

  // Intentionally copyable.
};

// Shape rank and dimensions are accessed through InferenceContext.
class Shape {
 private:
  Shape();
  Shape(const std::vector<DimensionHandle>& dims);
  ~Shape() {}

  const int32 rank_;
  const std::vector<DimensionHandle> dims_;

  friend class InferenceContext;

  TF_DISALLOW_COPY_AND_ASSIGN(Shape);
};

class ShapeHandle {
 public:
  ShapeHandle() {}

 private:
  ShapeHandle(const Shape* shape) { ptr_ = shape; }
  const Shape* operator->() { return ptr_; }
  bool IsSet() const { return ptr_ != nullptr; }
  bool SameHandle(ShapeHandle s) const { return ptr_ == s.ptr_; }

  const Shape* ptr_ = nullptr;

  friend class InferenceContext;
  friend class ShapeInferenceTest;
  friend class ShapeInferenceTestutil;

  // Intentionally copyable.
};

// Struct used to allow functions to take DimensionHandle or a dimension value.
// Not meant to be constructed directly.
struct DimensionOrConstant {
 public:
  // Intentionally not explicit.
  DimensionOrConstant(DimensionHandle dim);

  // val must be non-negative or InferenceContext::kUnknownDim.
  DimensionOrConstant(int64 val);

  // dim takes precedence. If dim != nullptr, val is ignored.
  DimensionHandle dim;
  int64 val;

 private:
  DimensionOrConstant();
};

// Note: This is experimental support for op shape inference in C++.  Shape
// inference functions are not ready to be implemented yet.
//
// An InferenceContext is created by the framework and passed to a shape
// inference function.  The shape inference function calls functions on the
// context, and should call set_output() to set the shape on all outputs.
//
// All Shape* and Dimension* returned by functions of InferenceContext are owned
// by the InferenceContext.
class InferenceContext {
 public:
  static constexpr int64 kUnknownDim = -1;
  static constexpr int32 kUnknownRank = -1;

  // <input_tensors> is NULL-padded to be the same size as <input_shapes>.
  //
  // REQUIRES: <node_def> is not NULL, and must outlive the InferenceContext.
  //
  // TODO(vrv): Remove 'input_shapes_string' once we can move the
  // creation of Shapes from strings out of this class (or hide it).
  InferenceContext(const NodeDef* node_def, const OpDef& op_def,
                   const std::vector<string>& input_shapes_string,
                   const std::vector<ShapeHandle>& input_shapes,
                   const std::vector<const Tensor*>& input_tensors);

  // <input_tensors> is NULL-padded to be the same size as <input_shapes>.
  //
  // REQUIRES: <node_def> is not NULL, and must outlive the InferenceContext.
  //
  // TODO(cwhipkey): Remove 'input_shapes_string' once we can move the creation
  // of Shapes from strings out of this class (or hide it).
  InferenceContext(const NodeDef* node_def, const OpDef& op_def,
                   const std::vector<string>& input_shapes_string,
                   const std::vector<TensorShapeProto>& input_shapes,
                   const std::vector<const Tensor*>& input_tensors);

  // This is a temporary constructor used for initial testing.
  //
  // TODO(cwhipkey): remove this temporary constructor.
  //
  // Each input shape describes the input shape as follows:
  // * "?" : the shape's rank and dimensions are unknown
  // * "[1,?,3]" : the shape's rank is known, and dimensions can be known or
  //               unknown (? for unknown #1 - multiple dimensions can be
  //               labeled with the same unknown number, and are deduplicated to
  //               the same Dimension*.
  //
  // <input_tensors> is NULL-padded to be the same size as <input_shapes>.
  //
  // REQUIRES: <node_def> is not NULL, and must outlive the InferenceContext.
  InferenceContext(const NodeDef* node_def, const OpDef& op_def,
                   const std::vector<string>& input_shapes,
                   const std::vector<const Tensor*>& input_tensors);

  ~InferenceContext();

  ShapeHandle input(int idx) const { return inputs_[idx]; }
  Status input(StringPiece input_name, std::vector<ShapeHandle>* output) const;
  int num_inputs() const { return inputs_.size(); }

  // Returns the input tensor at index <idx>, or nullptr if the input tensor is
  // not available at the time of shape inference.
  const Tensor* input_tensor(int idx) {
    // Mark that this idx was requested.
    requested_input_tensor_[idx] = true;
    return input_tensors_[idx];
  }

  // Returns true iff input_tensor(idx) was called by the shape function.
  bool requested_input_tensor(int idx) const {
    return requested_input_tensor_[idx];
  }

  void set_input_tensors(const std::vector<const Tensor*>& input_tensors) {
    input_tensors_ = input_tensors;
  }

  void set_output(int idx, ShapeHandle shape) { outputs_[idx] = shape; }
  Status set_output(StringPiece output_name,
                    const std::vector<ShapeHandle>& shapes);

  int num_outputs() const { return outputs_.size(); }
  ShapeHandle output(int idx) const { return outputs_[idx]; }
  Status output(StringPiece output_name,
                std::vector<ShapeHandle>* output) const;

  // idx can be negative for an offset from end of dimensions.
  // idx must be in the range [-1 * s.rank, s.rank).
  DimensionHandle Dim(ShapeHandle s, int32 idx) {
    if (s->rank_ == kUnknownRank) {
      return UnknownDim();
    }
    if (idx < 0) {
      return s->dims_[s->dims_.size() + idx];
    }
    return s->dims_[idx];
  }
  int32 Rank(ShapeHandle s) { return s->rank_; }
  bool RankKnown(ShapeHandle s) { return Rank(s) != kUnknownRank; }
  inline int64 Value(DimensionOrConstant d) {
    return d.dim.IsSet() ? d.dim->value_ : d.val;
  }
  inline bool ValueKnown(DimensionOrConstant d) {
    return Value(d) != kUnknownDim;
  }

  // Returns true if the rank and all dimensions of the Shape are known.
  bool FullyDefined(ShapeHandle s);

  // Returns the total number of elements, or an unknown dimension for an
  // incomplete shape.
  DimensionHandle NumElements(ShapeHandle s);

  string DebugString(ShapeHandle s);
  string DebugString(DimensionHandle d);

  // If <shape> has rank <rank>, or its rank is unknown, return OK and return
  // the shape with asserted rank in <*out>. Otherwise return an error.
  //
  // Note that <*out> may be set to <shape>.
  Status WithRank(ShapeHandle shape, int32 rank,
                  ShapeHandle* out) TF_MUST_USE_RESULT;
  Status WithRankAtLeast(ShapeHandle shape, int32 rank,
                         ShapeHandle* out) TF_MUST_USE_RESULT;
  Status WithRankAtMost(ShapeHandle shape, int32 rank,
                        ShapeHandle* out) TF_MUST_USE_RESULT;

  // If <dim> has value <value>, or its value is unknown, returns OK and returns
  // the dimension with asserted value in <*out>. Otherwise returns an error.
  //
  // Note that <*out> may be set to <dim>.
  Status WithValue(DimensionHandle dim, int64 value,
                   DimensionHandle* out) TF_MUST_USE_RESULT;

  // Merges <in0> and <in1> and returns the merged shape in <*out>. If <in0> and
  // <in1> are incompatible in rank, or in the value of any dimension, returns
  // an error.
  //
  // Note that <*out> may be set to <in0> or <in1>.
  Status Merge(ShapeHandle in0, ShapeHandle in1,
               ShapeHandle* out) TF_MUST_USE_RESULT;

  // Asserts that <s>'s rank >= <prefix>'s rank, and the first
  // <prefix.rank> dimensions of <s> are compatible with the dimensions of
  // <prefix>.
  // Returns the merged results in <*s_out> and <*prefix_out>.
  Status MergePrefix(ShapeHandle s, ShapeHandle prefix, ShapeHandle* s_out,
                     ShapeHandle* prefix_out) TF_MUST_USE_RESULT;

  // Merges <d0> and <d1> and returns the merged dimension in <*out>. If <d0>
  // and <d1> have incompatible values, returns an error.
  //
  // Note that <*out> may be set to <d0> or <d1>.
  Status Merge(DimensionHandle d0, DimensionHandle d1,
               DimensionHandle* out) TF_MUST_USE_RESULT;

  // Returns in <*out> a sub-shape of <s> with dimensions [start:].
  // <start> can be negative to index from the end of the shape. If <start> >
  // rank of <s>, then an empty subshape is returned.
  Status Subshape(ShapeHandle s, int64 start,
                  ShapeHandle* out) TF_MUST_USE_RESULT;

  // Returns in <*out> a sub-shape of <s>, with dimensions [start:end].
  // <start> and <end> can be negative, to index from the end of the shape.
  // <start> and <end> are set to the rank of <s> if > rank of <s>.
  Status Subshape(ShapeHandle s, int64 start, int64 end,
                  ShapeHandle* out) TF_MUST_USE_RESULT;

  // Returns in <*out> the result of appending the dimensions of <s2> to those
  // of <s1>.
  Status Concatenate(ShapeHandle s1, ShapeHandle s2,
                     ShapeHandle* out) TF_MUST_USE_RESULT;

  // Returns in <out> the shape from replacing <s.dim[dim_index]> with
  // <new_dim>.
  Status ReplaceDim(ShapeHandle s, int dim_index, DimensionHandle new_dim,
                    ShapeHandle* out) TF_MUST_USE_RESULT;

  // Returns a new shape with the given dims. The returned value is owned by
  // this context.
  ShapeHandle MakeShape(const std::vector<DimensionHandle>& dims);
  ShapeHandle MakeShape(std::initializer_list<DimensionOrConstant> dims);

  // Returns a new unknown shape.
  ShapeHandle UnknownShape();

  // Returns a shape with specified rank but unknown dims.
  ShapeHandle UnknownShapeOfRank(int32 rank);

  // Returns a new shape of zero dimensions.
  ShapeHandle Scalar();

  // Returns a new shape of one dimension.
  ShapeHandle Vector(DimensionOrConstant dim);

  // Returns a new shape of two dimensions.
  ShapeHandle Matrix(DimensionOrConstant dim1, DimensionOrConstant dim2);

  // Returns in <out> a new shape whose dimension sizes come from input tensor
  // <input_idx>. The tensor must be a 1-dimensional int32 or int64 tensor.  If
  // the input tensor is NULL, then an unknown shape is returned.
  Status MakeShapeFromShapeTensor(int input_idx, ShapeHandle* out);

  // Returns in <out> a new shape corresponding to <proto>.
  Status MakeShapeFromShapeProto(const TensorShapeProto& proto,
                                 ShapeHandle* out);

  // Returns a new dimension of the given size.  The returned value is owned by
  // this context.
  inline DimensionHandle MakeDim(DimensionOrConstant d) {
    if (d.dim.IsSet()) {
      return d.dim;
    } else {
      all_dims_.push_back(new Dimension(d.val));
      return all_dims_.back();
    }
  }
  inline DimensionHandle UnknownDim() { return MakeDim(kUnknownDim); }

  // Returns a new dimension whose value is given by a scalar input tensor.
  // The input tensor must be in host memory, since it is dereferenced to get
  // the value.
  Status MakeDimForScalarInput(int idx, DimensionHandle* out);

  // Look up the attr for the NodeDef being evaluated with name attr_name and
  // set *value to its value.  If no attr with attr_name is found in def(), or
  // the attr does not have a matching type, a non-ok status will be returned.
  template <class T>
  Status GetAttr(StringPiece attr_name, T* value) const;

  // Returns in <out> the result of dividing <dividend> by <divisor>.
  // Returns an error if <divisor>  is not positive or if <evenly_divisible>
  // and <divisor> does not evenly divide <dividend>.
  Status Divide(DimensionHandle dividend, int64 divisor, bool evenly_divisible,
                DimensionHandle* out);

  // Returns in <out> the sum of <first> and <second>.
  Status Add(DimensionHandle first, DimensionOrConstant second,
             DimensionHandle* out);

  // Returns in <out> the dimension that is <first> minus <second>.
  Status Subtract(DimensionHandle first, DimensionOrConstant second,
                  DimensionHandle* out);

  // Returns in <out> the product of <first> and <second>.
  Status Multiply(DimensionHandle first, DimensionOrConstant second,
                  DimensionHandle* out);

  // Returns in <out> the minimum of <first> and <second>. If either <first> or
  // <second> is zero the results is zero. Otherwise, if either <first> or
  // <second> is unknown the results is unknown.
  Status Min(DimensionHandle first, DimensionOrConstant second,
             DimensionHandle* out);

  // Returns in <out> the maximum of <first> and <second>. If either <first> or
  // <second> is unknown the results is unknown.
  Status Max(DimensionHandle first, DimensionOrConstant second,
             DimensionHandle* out);

  Status construction_status() const { return construction_status_; }

  // Validates the 3 component tensors of a sparse tensor have the proper
  // shapes. This mimics SparseTensor.__init__ in python/framework/ops.py.
  Status ValidateSparseTensor(ShapeHandle indices_shape,
                              ShapeHandle values_shape,
                              ShapeHandle shape_shape) {
    // Validate ranks.
    ShapeHandle unused_shape;
    TF_RETURN_IF_ERROR(WithRank(indices_shape, 2, &unused_shape));
    TF_RETURN_IF_ERROR(WithRank(values_shape, 1, &unused_shape));
    TF_RETURN_IF_ERROR(WithRank(shape_shape, 1, &unused_shape));

    // Number of elements in indices and values must match.
    DimensionHandle num_index_elements_dim = Dim(indices_shape, 0);
    if (ValueKnown(num_index_elements_dim)) {
      DimensionHandle num_values_elements_dim = Dim(values_shape, 0);
      if (ValueKnown(num_values_elements_dim)) {
        int64 num_index_elements = Value(num_index_elements_dim);
        int64 num_values_elements = Value(num_values_elements_dim);
        if (num_index_elements != num_values_elements) {
          return errors::InvalidArgument(
              "Number of elements in index (", num_index_elements,
              ") and values (", num_values_elements, ") do not match.");
        }
      }
    }

    // Rank embedded in indices must match shape.
    DimensionHandle index_rank_dim = Dim(indices_shape, 1);
    if (ValueKnown(index_rank_dim)) {
      DimensionHandle shape_rank_dim = Dim(shape_shape, 0);
      if (ValueKnown(shape_rank_dim)) {
        int64 index_rank = Value(index_rank_dim);
        int32 shape_rank = Value(shape_rank_dim);
        if (index_rank != shape_rank) {
          return errors::InvalidArgument("Index rank (", index_rank,
                                         ") and shape rank (", shape_rank,
                                         ") do not match.");
        }
      }
    }

    return Status::OK();
  }

 private:
  // Shared initialization across the two constructors.  Remove
  // once we get rid of one of them.
  void PreInputInit(const OpDef& op_def,
                    const std::vector<const Tensor*>& input_tensors);
  void PostInputInit();

  // Returns a shape from 'shape_string'.
  Status MakeShapeFromString(const string& shape_string, ShapeHandle* output);

  DimensionHandle GetDimension(const DimensionOrConstant& d);

  Status ReturnUnknownShape(ShapeHandle* out) {
    *out = UnknownShape();
    return Status::OK();
  }
  Status ReturnCreatedShape(const std::vector<DimensionHandle>& dims,
                            ShapeHandle* out) {
    *out = MakeShape(dims);
    return Status::OK();
  }

  std::vector<Shape*> all_shapes_;    // values are owned.
  std::vector<Dimension*> all_dims_;  // values are owned.

  // inputs_ and outputs_ refer to values from all_shapes_.
  std::vector<ShapeHandle> inputs_;
  std::vector<const Tensor*> input_tensors_;
  std::vector<bool> requested_input_tensor_;
  std::vector<ShapeHandle> outputs_;

  const NodeDef& node_def_;
  NameRangeMap input_name_map_;
  NameRangeMap output_name_map_;

  // An error set during construction. TODO(cwhipkey): remove when test
  // constructor is removed.
  Status construction_status_;

  TF_DISALLOW_COPY_AND_ASSIGN(InferenceContext);
};

// -----------------------------------------------------------------------------
// Template and inline method implementations, please ignore

inline Dimension::Dimension() : value_(InferenceContext::kUnknownDim) {}
inline Dimension::Dimension(int64 value) : value_(value) {
  DCHECK(value >= 0 || value == InferenceContext::kUnknownDim)
      << "Dimension must be non-negative or equal to "
         "InferenceContext::kUnknownDim but got"
      << value;
}

inline Shape::Shape() : rank_(InferenceContext::kUnknownRank) {}
inline Shape::Shape(const std::vector<DimensionHandle>& dims)
    : rank_(dims.size()), dims_(dims) {}

inline DimensionOrConstant::DimensionOrConstant(DimensionHandle dim)
    : dim(dim) {
  DCHECK(dim.IsSet()) << "Internal error: Got nullptr for Dimension.";
}

inline DimensionOrConstant::DimensionOrConstant(int64 val) : val(val) {
  DCHECK(val >= 0 || val == InferenceContext::kUnknownDim)
      << "Dimension must be non-negative or equal to "
         "InferenceContext::kUnknownDim but got"
      << val;
}

template <class T>
Status InferenceContext::GetAttr(StringPiece attr_name, T* value) const {
  return GetNodeAttr(node_def_, attr_name, value);
}

}  // namespace shape_inference
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_FRAMEWORK_SHAPE_INFERENCE_H_
