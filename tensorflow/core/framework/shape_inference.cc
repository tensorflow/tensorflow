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
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace shape_inference {

constexpr int32 InferenceContext::kUnknownRank;
constexpr int64 InferenceContext::kUnknownDim;

InferenceContext::InferenceContext(
    const NodeDef* node_def, const OpDef& op_def,
    const std::vector<string>& input_shapes,
    const std::vector<const Tensor*>& input_tensors)
    : node_def_(*CHECK_NOTNULL(node_def)) {
  PreInputInit(op_def, input_tensors);

  for (const string& spec : input_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromString(spec, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    inputs_.push_back(shape);
  }

  PostInputInit();
}

InferenceContext::InferenceContext(
    const NodeDef* node_def, const OpDef& op_def,
    const std::vector<string>& input_shapes_string,
    const std::vector<TensorShapeProto>& input_shapes,
    const std::vector<const Tensor*>& input_tensors)
    : node_def_(*CHECK_NOTNULL(node_def)) {
  PreInputInit(op_def, input_tensors);
  if (!construction_status_.ok()) return;
  for (const TensorShapeProto& p : input_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromShapeProto(p, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    inputs_.push_back(shape);
  }
  PostInputInit();
}

InferenceContext::InferenceContext(
    const NodeDef* node_def, const OpDef& op_def,
    const std::vector<string>& input_shapes_string,
    const std::vector<ShapeHandle>& input_shapes,
    const std::vector<const Tensor*>& input_tensors)
    : node_def_(*CHECK_NOTNULL(node_def)) {
  PreInputInit(op_def, input_tensors);
  if (!construction_status_.ok()) return;
  inputs_ = input_shapes;
  PostInputInit();
}

InferenceContext::~InferenceContext() {
  for (auto* s : all_shapes_) delete s;
  for (auto* d : all_dims_) delete d;
}

Status InferenceContext::set_output(StringPiece output_name,
                                    const std::vector<ShapeHandle>& shapes) {
  const auto result = output_name_map_.find(output_name.ToString());
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    const int start = result->second.first;
    const int size = result->second.second - start;
    if (size != shapes.size()) {
      errors::InvalidArgument("Must have exactly ", shapes.size(), " shapes.");
    }
    for (int i = 0; i < size; ++i) {
      outputs_[i + start] = shapes[i];
    }
  }
  return Status::OK();
}

Status InferenceContext::input(StringPiece input_name,
                               std::vector<ShapeHandle>* output) const {
  const auto result = input_name_map_.find(input_name.ToString());
  if (result == input_name_map_.end()) {
    return errors::InvalidArgument("Unknown input name: ", input_name);
  } else {
    output->clear();
    for (int i = result->second.first; i < result->second.second; ++i) {
      output->push_back(inputs_[i]);
    }
  }
  return Status::OK();
}

Status InferenceContext::output(StringPiece output_name,
                                std::vector<ShapeHandle>* output) const {
  const auto result = output_name_map_.find(output_name.ToString());
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    output->clear();
    for (int i = result->second.first; i < result->second.second; ++i) {
      output->push_back(outputs_[i]);
    }
  }
  return Status::OK();
}

void InferenceContext::PreInputInit(
    const OpDef& op_def, const std::vector<const Tensor*>& input_tensors) {
  input_tensors_ = input_tensors;

  construction_status_ =
      NameRangesForNode(node_def_, op_def, &input_name_map_, &output_name_map_);
  if (!construction_status_.ok()) return;

  int num_outputs = 0;
  for (const auto& e : output_name_map_) {
    num_outputs = std::max(num_outputs, e.second.second);
  }
  for (int i = 0; i < num_outputs; ++i) {
    outputs_.push_back(nullptr);
  }
}

void InferenceContext::PostInputInit() {
  int num_inputs_from_node_def = 0;
  for (const auto& e : input_name_map_) {
    num_inputs_from_node_def =
        std::max(num_inputs_from_node_def, e.second.second);
  }

  if (inputs_.size() != num_inputs_from_node_def) {
    construction_status_ = errors::InvalidArgument(
        "Wrong number of inputs passed: ", inputs_.size(), " while ",
        num_inputs_from_node_def, " expected based on NodeDef");
    return;
  }

  CHECK_LE(input_tensors_.size(), inputs_.size());
  input_tensors_.resize(inputs_.size());
  requested_input_tensor_.resize(inputs_.size());
}

bool InferenceContext::FullyDefined(ShapeHandle s) {
  if (!RankKnown(s)) return false;
  for (int i = 0; i < Rank(s); ++i) {
    if (!ValueKnown(Dim(s, i))) return false;
  }
  return true;
}

DimensionHandle InferenceContext::NumElements(ShapeHandle s) {
  const auto rank = Rank(s);
  if (rank == kUnknownRank) return UnknownDim();
  int64 size = 1;
  for (int i = 0; i < rank; ++i) {
    int64 dim_val = Value(Dim(s, i));
    if (dim_val == kUnknownDim) return UnknownDim();
    size *= dim_val;
  }
  return MakeDim(size);
}

string InferenceContext::DebugString(ShapeHandle s) {
  if (RankKnown(s)) {
    std::vector<string> vals;
    for (auto d : s->dims_) vals.push_back(DebugString(d));
    return strings::StrCat("[", str_util::Join(vals, ","), "]");
  } else {
    return "?";
  }
}

string InferenceContext::DebugString(DimensionHandle d) {
  return ValueKnown(d) ? strings::StrCat(Value(d)) : "?";
}

Status InferenceContext::WithRank(ShapeHandle shape, int32 rank,
                                  ShapeHandle* out) {
  const int32 existing = Rank(shape);
  if (existing == rank) {
    *out = shape;
    return Status::OK();
  }
  if (existing == kUnknownRank) {
    std::vector<DimensionHandle> dims;
    dims.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      all_dims_.push_back(new Dimension());
      dims.push_back(all_dims_.back());
    }
    all_shapes_.push_back(new Shape(dims));
    *out = all_shapes_.back();
    return Status::OK();
  }
  *out = nullptr;
  return errors::InvalidArgument("Shape must be rank ", rank, " but is rank ",
                                 existing);
}

Status InferenceContext::WithRankAtLeast(ShapeHandle shape, int32 rank,
                                         ShapeHandle* out) {
  const int32 existing = Rank(shape);
  if (existing >= rank) {
    *out = shape;
    return Status::OK();
  }
  if (existing == kUnknownRank) {
    return ReturnUnknownShape(out);
  }
  *out = nullptr;
  return errors::InvalidArgument("Shape must be at least rank ", rank,
                                 " but is rank ", existing);
}

Status InferenceContext::WithRankAtMost(ShapeHandle shape, int32 rank,
                                        ShapeHandle* out) {
  const int32 existing = Rank(shape);
  if (existing == kUnknownRank) {
    return ReturnUnknownShape(out);
  }
  if (existing <= rank) {
    *out = shape;
    return Status::OK();
  }
  *out = nullptr;
  return errors::InvalidArgument("Shape must be at most rank ", rank,
                                 " but is rank ", existing);
}

Status InferenceContext::WithValue(DimensionHandle dim, int64 value,
                                   DimensionHandle* out) {
  const int64 existing = Value(dim);
  if (existing == value) {
    *out = dim;
    return Status::OK();
  }
  if (existing == kUnknownDim) {
    all_dims_.push_back(new Dimension(value));
    *out = all_dims_.back();
    return Status::OK();
  }
  *out = nullptr;
  return errors::InvalidArgument("Dimension must be ", value, " but is ",
                                 existing);
}

Status InferenceContext::Merge(DimensionHandle d0, DimensionHandle d1,
                               DimensionHandle* out) {
  if (d0.SameHandle(d1) || !ValueKnown(d1)) {
    *out = d0;
    return Status::OK();
  } else if (!ValueKnown(d0)) {
    *out = d1;
    return Status::OK();
  } else if (Value(d0) == Value(d1)) {
    *out = d0;
    return Status::OK();
  } else {
    *out = nullptr;
    return errors::InvalidArgument("Dimensions must be equal, but are ",
                                   Value(d0), " and ", Value(d1));
  }
}

Status InferenceContext::MergePrefix(ShapeHandle s, ShapeHandle prefix,
                                     ShapeHandle* s_out,
                                     ShapeHandle* prefix_out) {
  *s_out = *prefix_out = nullptr;
  if (!RankKnown(prefix) || !RankKnown(s)) {
    *s_out = s;
    *prefix_out = prefix;
    return Status::OK();
  }
  const int32 rank = Rank(prefix);
  TF_RETURN_IF_ERROR(WithRankAtLeast(s, rank, &s));

  // Merge the prefix dims and create the new output shapes.
  std::vector<DimensionHandle> dims;
  dims.resize(rank);
  for (int i = 0; i < rank; ++i) {
    TF_RETURN_IF_ERROR(Merge(Dim(s, i), Dim(prefix, i), &dims[i]));
  }
  *prefix_out = MakeShape(dims);
  for (int i = rank; i < Rank(s); ++i) dims.push_back(Dim(s, i));
  *s_out = MakeShape(dims);
  return Status::OK();
}

Status InferenceContext::Merge(ShapeHandle s0, ShapeHandle s1,
                               ShapeHandle* out) {
  if (s0.SameHandle(s1) || !RankKnown(s1)) {
    *out = s0;
    return Status::OK();
  } else if (!RankKnown(s0)) {
    *out = s1;
    return Status::OK();
  }

  const int32 rank = Rank(s0);
  if (rank != Rank(s1)) {
    *out = nullptr;
    return errors::InvalidArgument("Shapes must be equal rank, but are ", rank,
                                   " and ", Rank(s1));
  }

  bool return_s0 = true;
  bool return_s1 = true;
  for (int i = 0; i < rank; ++i) {
    auto d0 = Dim(s0, i);
    auto d1 = Dim(s1, i);
    if (d0.SameHandle(d1)) continue;

    auto v0 = Value(d0);
    auto v1 = Value(d1);
    if (v0 == kUnknownDim) {
      if (v1 != kUnknownDim) {
        return_s0 = false;
      }
    } else if (v1 == kUnknownDim) {
      return_s1 = false;
    } else if (v0 != v1) {
      *out = nullptr;
      return errors::InvalidArgument("Dimension ", i,
                                     " in both shapes must be equal, but are ",
                                     Value(d0), " and ", Value(d1));
    }
  }
  if (return_s0 || return_s1) {
    *out = return_s0 ? s0 : s1;
    return Status::OK();
  }

  // Merge dims.
  std::vector<DimensionHandle> dims(rank, nullptr);
  for (int i = 0; i < rank; ++i) {
    // Invariant for merge was checked earlier, so CHECK is ok.
    TF_CHECK_OK(Merge(Dim(s0, i), Dim(s1, i), &dims[i]));
  }
  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::Subshape(ShapeHandle s, int64 start,
                                  ShapeHandle* out) {
  return Subshape(s, start, std::numeric_limits<int64>::max() /* end */, out);
}

Status InferenceContext::Subshape(ShapeHandle s, int64 start_in, int64 end_in,
                                  ShapeHandle* out) {
  int64 start = start_in;
  int64 end = end_in;
  const int32 rank = Rank(s);
  if (start == 0 && ((RankKnown(s) && end >= rank) ||
                     end == std::numeric_limits<int64>::max())) {
    *out = s;
    return Status::OK();
  }
  if (!RankKnown(s)) {
    return ReturnUnknownShape(out);
  }

  if (start > rank) start = rank;
  if (end > rank) end = rank;
  if (start < 0) {
    start = rank + start;
    if (start < 0) {
      *out = nullptr;
      return errors::InvalidArgument("Subshape start out of bounds: ", start_in,
                                     ", for shape with rank ", rank);
    }
  }

  if (end < 0) {
    end = rank + end;
    if (end < 0) {
      *out = nullptr;
      return errors::InvalidArgument("Subshape end out of bounds: ", end_in,
                                     ", for shape with rank ", rank);
    }
  }
  if (start > end) {
    *out = nullptr;
    return errors::InvalidArgument(
        "Subshape must have computed start <= end, but is ", start, " and ",
        end, " (computed from start ", start_in, " and end ", end_in,
        " over shape with rank ", rank, ")");
  }
  std::vector<DimensionHandle> dims;
  dims.reserve(end - start);
  for (int i = start; i < end; ++i) {
    dims.push_back(Dim(s, i));
  }
  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::Concatenate(ShapeHandle s1, ShapeHandle s2,
                                     ShapeHandle* out) {
  if (!RankKnown(s1) || !RankKnown(s2)) {
    return ReturnUnknownShape(out);
  }
  const int32 s1_rank = Rank(s1);
  const int32 s2_rank = Rank(s2);
  const int32 rank = s1_rank + s2_rank;
  std::vector<DimensionHandle> dims;
  dims.reserve(rank);
  for (int i = 0; i < s1_rank; ++i) dims.push_back(Dim(s1, i));
  for (int i = 0; i < s2_rank; ++i) dims.push_back(Dim(s2, i));
  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::ReplaceDim(ShapeHandle s, int dim_index_in,
                                    DimensionHandle new_dim, ShapeHandle* out) {
  if (!RankKnown(s)) {
    return ReturnUnknownShape(out);
  }
  int dim_index = dim_index_in;
  if (dim_index < 0) {
    dim_index = s->dims_.size() + dim_index;
  }
  if (!FastBoundsCheck(dim_index, s->dims_.size())) {
    *out = nullptr;
    return errors::InvalidArgument("Out of range dim_index ", dim_index_in,
                                   " for shape with ", s->dims_.size(),
                                   " dimensions");
  }
  std::vector<DimensionHandle> dims(s->dims_);
  dims[dim_index] = new_dim;
  return ReturnCreatedShape(dims, out);
}

ShapeHandle InferenceContext::MakeShape(
    const std::vector<DimensionHandle>& dims) {
  all_shapes_.push_back(new Shape(dims));
  return all_shapes_.back();
}

ShapeHandle InferenceContext::MakeShape(
    std::initializer_list<DimensionOrConstant> dims) {
  std::vector<DimensionHandle> dims_actual;
  dims_actual.reserve(dims.size());
  for (const DimensionOrConstant& d : dims) {
    dims_actual.push_back(MakeDim(d));
  }
  return MakeShape(dims_actual);
}

ShapeHandle InferenceContext::UnknownShape() {
  all_shapes_.push_back(new Shape());
  return all_shapes_.back();
}

ShapeHandle InferenceContext::UnknownShapeOfRank(int32 rank) {
  std::vector<DimensionHandle> dims(rank);
  for (int32 i = 0; i < rank; ++i) {
    dims[i] = UnknownDim();
  }
  return MakeShape(dims);
}

ShapeHandle InferenceContext::Scalar() { return MakeShape({}); }

ShapeHandle InferenceContext::Vector(DimensionOrConstant dim) {
  return MakeShape({dim});
}

ShapeHandle InferenceContext::Matrix(DimensionOrConstant dim1,
                                     DimensionOrConstant dim2) {
  return MakeShape({dim1, dim2});
}

Status InferenceContext::MakeShapeFromShapeTensor(int input_idx,
                                                  ShapeHandle* out) {
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(WithRank(input(input_idx), 1, &input_shape));

  const Tensor* t = input_tensor(input_idx);
  if (t == nullptr) {
    // Shape tensor is not known, but if the shape of the shape tensor is then
    // the right number of unknown dims can be created.
    DimensionHandle shape_dim = Dim(input_shape, 0);
    if (!ValueKnown(shape_dim)) {
      return ReturnUnknownShape(out);
    }
    const auto num_dims = Value(shape_dim);
    std::vector<DimensionHandle> dims;
    for (int i = 0; i < num_dims; i++) dims.push_back(UnknownDim());
    return ReturnCreatedShape(dims, out);
  }

  if (t->shape().dims() != 1) {
    *out = nullptr;
    return errors::InvalidArgument("Input tensor must be rank 1, but was rank ",
                                   t->shape().dims());
  }
  std::vector<DimensionHandle> dims;
  if (t->dtype() == DataType::DT_INT32) {
    auto flat_t = t->flat<int32>();
    for (int i = 0; i < flat_t.size(); ++i) {
      dims.push_back(MakeDim(flat_t(i)));
    }
  } else if (t->dtype() == DataType::DT_INT64) {
    auto flat_t = t->flat<int64>();
    for (int i = 0; i < flat_t.size(); ++i) {
      dims.push_back(MakeDim(flat_t(i)));
    }
  } else {
    *out = nullptr;
    return errors::InvalidArgument(
        "Input tensor must be int32 or int64, but was ",
        DataTypeString(t->dtype()));
  }

  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::MakeShapeFromShapeProto(const TensorShapeProto& proto,
                                                 ShapeHandle* out) {
  *out = nullptr;
  TF_RETURN_IF_ERROR(PartialTensorShape::IsValidShape(proto));
  PartialTensorShape partial_shape(proto);
  if (partial_shape.dims() == -1) {
    return ReturnUnknownShape(out);
  }
  const int num_dims = partial_shape.dims();
  std::vector<DimensionHandle> dims;
  dims.reserve(partial_shape.dims());
  for (int i = 0; i < num_dims; ++i) {
    // -1 is unknown in proto and in InferenceContext, so this size can be
    // passed directly to MakeDim.
    dims.push_back(MakeDim(partial_shape.dim_size(i)));
  }
  return ReturnCreatedShape(dims, out);
}

// Returns a new dimension whose value is given by a scalar input tensor.
Status InferenceContext::MakeDimForScalarInput(int idx, DimensionHandle* out) {
  const Tensor* t = input_tensor(idx);
  if (t == nullptr) {
    *out = UnknownDim();
    return Status::OK();
  }
  const int rank = t->dims();
  if (rank != 0) {
    return errors::InvalidArgument("Input must be scalar but has rank ", rank);
  }

  int64 val;
  if (t->dtype() == DT_INT32) {
    val = t->scalar<int32>()();
  } else if (t->dtype() == DT_INT64) {
    val = t->scalar<int64>()();
  } else {
    return errors::InvalidArgument(
        "Scalar input for dim size must be int32 or int64");
  }
  if (val < 0) {
    return errors::InvalidArgument("Dimension size, given by scalar input ",
                                   idx, ", must be non-negative but is ", val);
  }
  *out = MakeDim(val);
  return Status::OK();
}

Status InferenceContext::Divide(DimensionHandle dividend, int64 divisor,
                                bool evenly_divisible, DimensionHandle* out) {
  if (divisor == 1) {
    *out = dividend;
  } else if (!ValueKnown(dividend)) {
    *out = UnknownDim();
  } else {
    const int64 v = Value(dividend);
    if (divisor <= 0) {
      return errors::InvalidArgument("Divisor must be positive but is ",
                                     divisor);
    }
    if (evenly_divisible && (v % divisor) != 0) {
      return errors::InvalidArgument(
          "Dimension size must be evenly divisible by ", divisor, " but is ",
          v);
    }
    *out = MakeDim(v / divisor);
  }
  return Status::OK();
}

Status InferenceContext::Add(DimensionHandle first, DimensionOrConstant second,
                             DimensionHandle* out) {
  const int64 first_value = Value(first);
  const int64 second_value = Value(second);
  // Special cases.
  if (first_value == 0) {
    *out = MakeDim(second);
  } else if (second_value == 0) {
    *out = MakeDim(first);
  } else if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    // Invariant: Both values are known and positive.
    const int64 sum = first_value + second_value;
    if (sum < 0) {
      return errors::InvalidArgument("Dimension size overflow from adding ",
                                     first_value, " and ", second_value);
    }
    *out = MakeDim(sum);
  }
  return Status::OK();
}

Status InferenceContext::Subtract(DimensionHandle first,
                                  DimensionOrConstant second,
                                  DimensionHandle* out) {
  const int64 first_value = Value(first);
  const int64 second_value = Value(second);
  // Special cases.
  if (second_value == 0) {
    *out = MakeDim(first);
  } else if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    // Invariant: Both values are known, first_value is non-negative, and
    // second_value is positive.
    if (first_value < second_value) {
      return errors::InvalidArgument(
          "Negative dimension size caused by subtracting ", second_value,
          " from ", first_value);
    }
    *out = MakeDim(first_value - second_value);
  }
  return Status::OK();
}

Status InferenceContext::Multiply(DimensionHandle first,
                                  DimensionOrConstant second,
                                  DimensionHandle* out) {
  const int64 first_value = Value(first);
  const int64 second_value = Value(second);
  // Special cases.
  if (first_value == 0) {
    *out = first;
  } else if (second_value == 0) {
    *out = MakeDim(second);
  } else if (first_value == 1) {
    *out = MakeDim(second);
  } else if (second_value == 1) {
    *out = first;
  } else if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    // Invariant: Both values are known and and greater than 1.
    const int64 product = first_value * second_value;
    if (product < 0) {
      return errors::InvalidArgument(
          "Negative dimension size caused by overflow when multiplying ",
          first_value, " and ", second_value);
    }
    *out = MakeDim(product);
  }
  return Status::OK();
}

Status InferenceContext::Min(DimensionHandle first, DimensionOrConstant second,
                             DimensionHandle* out) {
  const int64 first_value = Value(first);
  const int64 second_value = Value(second);
  if (first_value == 0) {
    *out = first;
  } else if (second_value == 0) {
    *out = MakeDim(second);
  } else if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    if (first_value <= second_value) {
      *out = first;
    } else {
      *out = MakeDim(second);
    }
  }
  return Status::OK();
}

Status InferenceContext::Max(DimensionHandle first, DimensionOrConstant second,
                             DimensionHandle* out) {
  const int64 first_value = Value(first);
  const int64 second_value = Value(second);
  if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    if (first_value >= second_value) {
      *out = first;
    } else {
      *out = MakeDim(second);
    }
  }
  return Status::OK();
}

Status InferenceContext::MakeShapeFromString(const string& spec,
                                             ShapeHandle* output) {
  if (spec == "?") {
    *output = UnknownShape();
    return Status::OK();
  }

  std::vector<DimensionHandle> dims;
  strings::Scanner scanner(spec);
  scanner.OneLiteral("[");
  while (scanner.Peek() != ']') {
    if (scanner.Peek() == '?') {
      scanner.OneLiteral("?");
      dims.push_back(UnknownDim());
    } else {
      scanner.RestartCapture().Many(strings::Scanner::DIGIT);
      StringPiece match;
      int64 dim_size = 0;
      CHECK(scanner.GetResult(nullptr, &match) &&
            strings::safe_strto64(match, &dim_size))
          << spec;
      dims.push_back(MakeDim(dim_size));
    }

    if (scanner.Peek() == ',') {
      scanner.OneLiteral(",");
    } else if (scanner.Peek() != ']') {
      return errors::InvalidArgument(
          "Invalid input spec (] not found in dim shape): ", spec);
    }
  }
  CHECK(scanner.OneLiteral("]").Eos().GetResult());
  *output = MakeShape(dims);

  return Status::OK();
}

}  // namespace shape_inference
}  // namespace tensorflow
