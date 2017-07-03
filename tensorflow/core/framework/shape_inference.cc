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

#include "tensorflow/core/framework/node_def.pb_text.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
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
    int graph_def_version, const NodeDef* node_def, const OpDef& op_def,
    const std::vector<TensorShapeProto>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<TensorShapeProto>& input_tensors_as_shapes,
    const std::vector<
        std::unique_ptr<std::vector<std::pair<TensorShapeProto, DataType>>>>&
        input_handle_shapes_and_types)
    : graph_def_version_(graph_def_version),
      node_def_(*CHECK_NOTNULL(node_def)) {
  std::vector<ShapeHandle> input_tensors_as_shape_handles;
  for (const TensorShapeProto& p : input_tensors_as_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromShapeProto(p, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    input_tensors_as_shape_handles.push_back(shape);
  }
  PreInputInit(op_def, input_tensors, input_tensors_as_shape_handles);
  if (!construction_status_.ok()) return;
  for (const TensorShapeProto& p : input_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromShapeProto(p, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    inputs_.push_back(shape);
  }
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>> handle_data(
      input_shapes.size());
  for (int i = 0; i < input_handle_shapes_and_types.size(); ++i) {
    const auto& v = input_handle_shapes_and_types[i];
    if (v == nullptr) {
      continue;
    }
    handle_data[i].reset(new std::vector<ShapeAndType>(v->size()));
    auto& new_v = *handle_data[i];
    for (int j = 0; j < v->size(); ++j) {
      const auto& p = (*v)[j];
      construction_status_.Update(
          MakeShapeFromShapeProto(p.first, &new_v[j].shape));
      if (!construction_status_.ok()) {
        return;
      }
      new_v[j].dtype = p.second;
    }
  }
  PostInputInit(std::move(handle_data));
}

// Same as above, but with PartialTensorShape instead of TensorShapeProto
InferenceContext::InferenceContext(
    int graph_def_version, const NodeDef* node_def, const OpDef& op_def,
    const std::vector<PartialTensorShape>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<PartialTensorShape>& input_tensors_as_shapes,
    const std::vector<
        std::unique_ptr<std::vector<std::pair<PartialTensorShape, DataType>>>>&
        input_handle_shapes_and_types)
    : graph_def_version_(graph_def_version),
      node_def_(*CHECK_NOTNULL(node_def)) {
  std::vector<ShapeHandle> input_tensors_as_shape_handles;
  for (const PartialTensorShape& p : input_tensors_as_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromPartialTensorShape(p, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    input_tensors_as_shape_handles.push_back(shape);
  }
  PreInputInit(op_def, input_tensors, input_tensors_as_shape_handles);
  if (!construction_status_.ok()) return;
  for (const PartialTensorShape& p : input_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromPartialTensorShape(p, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    inputs_.push_back(shape);
  }
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>> handle_data(
      input_shapes.size());
  for (int i = 0; i < input_handle_shapes_and_types.size(); ++i) {
    const auto& v = input_handle_shapes_and_types[i];
    if (v == nullptr) {
      continue;
    }
    handle_data[i].reset(new std::vector<ShapeAndType>(v->size()));
    auto& new_v = *handle_data[i];
    for (int j = 0; j < v->size(); ++j) {
      const auto& p = (*v)[j];
      construction_status_.Update(
          MakeShapeFromPartialTensorShape(p.first, &new_v[j].shape));
      if (!construction_status_.ok()) {
        return;
      }
      new_v[j].dtype = p.second;
    }
  }
  PostInputInit(std::move(handle_data));
}

InferenceContext::InferenceContext(
    int graph_def_version, const NodeDef* node_def, const OpDef& op_def,
    const std::vector<ShapeHandle>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<ShapeHandle>& input_tensors_as_shapes,
    std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
        input_handle_shapes_and_types)
    : graph_def_version_(graph_def_version),
      node_def_(*CHECK_NOTNULL(node_def)) {
  PreInputInit(op_def, input_tensors, input_tensors_as_shapes);
  if (!construction_status_.ok()) return;
  inputs_ = input_shapes;

  PostInputInit(std::move(input_handle_shapes_and_types));
}

InferenceContext::~InferenceContext() {}

Status InferenceContext::Run(
    const std::function<Status(shape_inference::InferenceContext* c)>& fn) {
  Status s = fn(this);
  if (!s.ok()) {
    return AttachContext(s);
  }
#ifndef NDEBUG
  for (int i = 0; i < num_outputs(); ++i) {
    DCHECK(output(i).IsSet())
        << i << " for " << node_def_.name() << " of type " << node_def_.op();
  }
#endif  // NDEBUG
  return s;
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
      return errors::InvalidArgument("Must have exactly ", shapes.size(),
                                     " shapes.");
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
    const OpDef& op_def, const std::vector<const Tensor*>& input_tensors,
    const std::vector<ShapeHandle>& input_tensors_as_shapes) {
  input_tensors_ = input_tensors;
  input_tensors_as_shapes_ = input_tensors_as_shapes;

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
  output_handle_shapes_and_types_.resize(num_outputs);
}

void InferenceContext::PostInputInit(
    std::vector<std::unique_ptr<std::vector<ShapeAndType>>> input_handle_data) {
  int num_inputs_from_node_def = 0;
  for (const auto& e : input_name_map_) {
    num_inputs_from_node_def =
        std::max(num_inputs_from_node_def, e.second.second);
  }

  // Allow passing empty shapes/dtypes to avoid changing every single test.
  if (input_handle_data.empty()) {
    input_handle_shapes_and_types_.resize(inputs_.size());
  } else {
    if (input_handle_data.size() != inputs_.size()) {
      construction_status_ = errors::InvalidArgument(
          "Wrong number of handle shapes passed; expected ", inputs_.size(),
          " got ", input_handle_data.size());
      return;
    }
    input_handle_shapes_and_types_ = std::move(input_handle_data);
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
  requested_input_tensor_as_partial_shape_.resize(inputs_.size());
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

string InferenceContext::DebugString() const {
  return strings::StrCat("InferenceContext for node: ",
                         ProtoDebugString(node_def_));
}

Status InferenceContext::WithRank(ShapeHandle shape, int64 rank,
                                  ShapeHandle* out) {
  if (rank > kint32max) {
    return errors::InvalidArgument("Rank cannot exceed kint32max");
  }
  const int32 existing = Rank(shape);
  if (existing == rank) {
    *out = shape;
    return Status::OK();
  }
  if (existing == kUnknownRank) {
    std::vector<DimensionHandle> dims;
    dims.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      dims.push_back(UnknownDim());
    }
    *out = shape_manager_.MakeShape(dims);
    return Status::OK();
  }
  *out = nullptr;

  return errors::InvalidArgument("Shape must be rank ", rank, " but is rank ",
                                 existing);
}

Status InferenceContext::WithRankAtLeast(ShapeHandle shape, int64 rank,
                                         ShapeHandle* out) {
  if (rank > kint32max) {
    return errors::InvalidArgument("Rank cannot exceed kint32max");
  }
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

Status InferenceContext::WithRankAtMost(ShapeHandle shape, int64 rank,
                                        ShapeHandle* out) {
  if (rank > kint32max) {
    return errors::InvalidArgument("Rank cannot exceed kint32max");
  }
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
    *out = MakeDim(value);
    return Status::OK();
  }
  *out = nullptr;
  return errors::InvalidArgument("Dimension must be ", value, " but is ",
                                 existing);
}

void InferenceContext::Relax(DimensionHandle d0, DimensionHandle d1,
                             DimensionHandle* out) {
  if (d0.SameHandle(d1)) {
    *out = d0;
  } else if (!ValueKnown(d0) || !ValueKnown(d1)) {
    *out = UnknownDim();
  } else if (Value(d0) == Value(d1)) {
    *out = d0;
  } else {
    *out = UnknownDim();
  }
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

void InferenceContext::Relax(ShapeHandle s0, ShapeHandle s1, ShapeHandle* out) {
  if (s0.SameHandle(s1)) {
    *out = s0;
    return;
  } else if (!RankKnown(s0) || !RankKnown(s1)) {
    *out = UnknownShape();
    return;
  }

  const int32 rank = Rank(s0);
  if (rank != Rank(s1)) {
    *out = UnknownShape();
    return;
  }

  bool return_s0 = true;
  for (int i = 0; i < rank; ++i) {
    auto d0 = Dim(s0, i);
    auto d1 = Dim(s1, i);
    if (d0.SameHandle(d1)) continue;

    auto v0 = Value(d0);
    auto v1 = Value(d1);
    if (v0 == kUnknownDim || v1 == kUnknownDim || v0 != v1) {
      return_s0 = false;
      break;
    }
  }
  if (return_s0) {
    *out = s0;
    return;
  }

  // Relax dims.
  std::vector<DimensionHandle> dims(rank);
  for (int i = 0; i < rank; ++i) {
    // Invariant for relax was checked earlier, so CHECK is ok.
    Relax(Dim(s0, i), Dim(s1, i), &dims[i]);
  }
  *out = MakeShape(dims);
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

Status InferenceContext::ReplaceDim(ShapeHandle s, int64 dim_index_in,
                                    DimensionHandle new_dim, ShapeHandle* out) {
  if (!RankKnown(s)) {
    return ReturnUnknownShape(out);
  }
  int64 dim_index = dim_index_in;
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
  return shape_manager_.MakeShape(dims);
}

ShapeHandle InferenceContext::MakeShape(
    std::initializer_list<DimensionOrConstant> dims) {
  std::vector<DimensionHandle> dims_actual;
  dims_actual.reserve(dims.size());
  for (const DimensionOrConstant& d : dims) {
    dims_actual.push_back(MakeDim(d));
  }

  return shape_manager_.MakeShape(dims_actual);
}

ShapeHandle InferenceContext::UnknownShape() {
  return shape_manager_.UnknownShape();
}

ShapeHandle InferenceContext::UnknownShapeOfRank(int64 rank) {
  CHECK_LE(rank, kint32max) << "rank must be less than kint32max";
  if(rank == kUnknownRank) {
    return UnknownShape();
  }
  CHECK_GE(rank, 0) << "rank must not be negative";
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

  requested_input_tensor_as_partial_shape_[input_idx] = true;
  if (input_idx < input_tensors_as_shapes_.size() &&
      input_tensors_as_shapes_[input_idx].IsSet() &&
      RankKnown(input_tensors_as_shapes_[input_idx])) {
    *out = input_tensors_as_shapes_[input_idx];
    return Status::OK();
  }

  return MakeShapeFromTensor(input_tensor(input_idx), input_shape, out);
}

Status InferenceContext::MakeShapeFromTensor(const Tensor* t,
                                             ShapeHandle tensor_shape,
                                             ShapeHandle* out) {
  if (t == nullptr) {
    // Shape tensor is not known, but if the shape of the shape tensor is then
    // the right number of unknown dims can be created.
    DimensionHandle shape_dim = Dim(tensor_shape, 0);
    if (!ValueKnown(shape_dim)) {
      return ReturnUnknownShape(out);
    }
    const auto num_dims = Value(shape_dim);
    std::vector<DimensionHandle> dims;
    dims.reserve(num_dims);
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
      const int32 val = flat_t(i);
      if (val < -1) {
        return errors::InvalidArgument(
            "Invalid value in tensor used for shape: ", val);
      }
      // -1 will become an unknown dim.
      dims.push_back(MakeDim(val));
    }
  } else if (t->dtype() == DataType::DT_INT64) {
    auto flat_t = t->flat<int64>();
    for (int i = 0; i < flat_t.size(); ++i) {
      const int64 val = flat_t(i);
      if (val < -1) {
        return errors::InvalidArgument(
            "Invalid value in tensor used for shape: ", val);
      }
      // -1 will become an unknown dim.
      dims.push_back(MakeDim(val));
    }
  } else {
    *out = nullptr;
    return errors::InvalidArgument(
        "Input tensor must be int32 or int64, but was ",
        DataTypeString(t->dtype()));
  }

  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::MakeShapeFromPartialTensorShape(
    const PartialTensorShape& partial_shape, ShapeHandle* out) {
  *out = nullptr;
  if (partial_shape.dims() == -1) {
    return ReturnUnknownShape(out);
  }
  const int num_dims = partial_shape.dims();
  std::vector<DimensionHandle> dims(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    // -1 is unknown in PartialTensorShape and in InferenceContext, so this size
    // can be passed directly to MakeDim.
    dims[i] = MakeDim(partial_shape.dim_size(i));
  }
  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::MakeShapeFromTensorShape(const TensorShape& shape,
                                                  ShapeHandle* out) {
  return MakeShapeFromPartialTensorShape(PartialTensorShape(shape.dim_sizes()),
                                         out);
}

Status InferenceContext::MakeShapeFromShapeProto(const TensorShapeProto& proto,
                                                 ShapeHandle* out) {
  *out = nullptr;
  TF_RETURN_IF_ERROR(PartialTensorShape::IsValidShape(proto));
  PartialTensorShape partial_shape(proto);
  return MakeShapeFromPartialTensorShape(partial_shape, out);
}

Status InferenceContext::GetScalarFromTensor(const Tensor* t, int64* val) {
  // Caller must ensure that <t> is not NULL.
  const int rank = t->dims();
  if (rank != 0) {
    return errors::InvalidArgument("Input must be scalar but has rank ", rank);
  }

  if (t->dtype() == DT_INT32) {
    *val = t->scalar<int32>()();
    return Status::OK();
  } else if (t->dtype() == DT_INT64) {
    *val = t->scalar<int64>()();
    return Status::OK();
  } else {
    return errors::InvalidArgument(
        "Scalar input for dim size must be int32 or int64");
  }
}

// Returns a new dimension whose value is given by a scalar input tensor.
Status InferenceContext::MakeDimForScalarInput(int idx, DimensionHandle* out) {
  int64 val;
  const Tensor* t = input_tensor(idx);
  if (t == nullptr) {
    *out = UnknownDim();
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(GetScalarFromTensor(t, &val));
  if (val < 0) {
    return errors::InvalidArgument("Dimension size, given by scalar input ",
                                   idx, ", must be non-negative but is ", val);
  }
  *out = MakeDim(val);
  return Status::OK();
}

Status InferenceContext::MakeDimForScalarInputWithNegativeIndexing(
    int idx, int input_rank, DimensionHandle* out) {
  int64 val;
  const Tensor* t = input_tensor(idx);
  if (t == nullptr) {
    *out = UnknownDim();
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(GetScalarFromTensor(t, &val));
  if (val < 0) {
    if (input_rank < 0) {
      *out = UnknownDim();
      return Status::OK();
    } else if (val + input_rank < 0) {
      return errors::InvalidArgument("Dimension size, given by scalar input ",
                                     val, " must be in range [-", input_rank,
                                     ", ", input_rank, ")");
    } else {
      val += input_rank;
    }
  } else if (input_rank >= 0 && val >= input_rank) {
    return errors::InvalidArgument("Dimension size, given by scalar input ",
                                   val, " must be in range [-", input_rank,
                                   ", ", input_rank, ")");
  }
  *out = MakeDim(val);
  return Status::OK();
}

Status InferenceContext::Divide(DimensionHandle dividend,
                                DimensionOrConstant divisor,
                                bool evenly_divisible, DimensionHandle* out) {
  const int64 divisor_value = Value(divisor);
  if (divisor_value == 1) {
    *out = dividend;
  } else if (!ValueKnown(dividend) ||
             (divisor.dim.IsSet() && !ValueKnown(divisor.dim))) {
    *out = UnknownDim();
  } else {
    const int64 v = Value(dividend);
    if (divisor_value <= 0) {
      return errors::InvalidArgument("Divisor must be positive but is ",
                                     divisor_value);
    }
    if (evenly_divisible && (v % divisor_value) != 0) {
      return errors::InvalidArgument(
          "Dimension size must be evenly divisible by ", divisor_value,
          " but is ", v);
    }
    *out = MakeDim(v / divisor_value);
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
    // Invariant: Both values are known and greater than 1.
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

Status InferenceContext::AttachContext(const Status& status) {
  std::vector<string> input_shapes;
  for (const ShapeHandle& input_shape : inputs_) {
    input_shapes.emplace_back(DebugString(input_shape));
  }

  // Add information about the input tensors and partial tensor shapes used.
  std::vector<string> input_from_tensors_str;
  std::vector<string> input_from_tensors_as_shape_str;
  for (int i = 0; i < inputs_.size(); ++i) {
    if (requested_input_tensor_as_partial_shape_[i] &&
        i < input_tensors_as_shapes_.size() &&
        input_tensors_as_shapes_[i].IsSet() &&
        RankKnown(input_tensors_as_shapes_[i])) {
      input_from_tensors_as_shape_str.push_back(strings::StrCat(
          "input[", i, "] = ", DebugString(input_tensors_as_shapes_[i])));
    } else if (requested_input_tensor_[i] && i < input_tensors_.size() &&
               input_tensors_[i] != nullptr) {
      input_from_tensors_str.push_back(strings::StrCat(
          "input[", i, "] = <",
          input_tensors_[i]->SummarizeValue(256 /* max_values */), ">"));
    }
  }

  string error_context = strings::StrCat(
      " for '", node_def_.name(), "' (op: '", node_def_.op(),
      "') with input shapes: ", str_util::Join(input_shapes, ", "));
  if (!input_from_tensors_str.empty()) {
    strings::StrAppend(&error_context, " and with computed input tensors: ",
                       str_util::Join(input_from_tensors_str, ", "));
  }
  if (!input_from_tensors_as_shape_str.empty()) {
    strings::StrAppend(&error_context,
                       " and with input tensors computed as partial shapes: ",
                       str_util::Join(input_from_tensors_as_shape_str, ","));
  }

  strings::StrAppend(&error_context, ".");
  return Status(status.code(),
                strings::StrCat(status.error_message(), error_context));
}

bool InferenceContext::MergeHandleShapesAndTypes(
    const std::vector<ShapeAndType>& shapes_and_types,
    std::vector<ShapeAndType>* to_update) {
  if (shapes_and_types.size() != to_update->size()) {
    return false;
  }
  std::vector<ShapeAndType> new_values(shapes_and_types.size());
  bool refined = false;
  for (int i = 0; i < shapes_and_types.size(); ++i) {
    const ShapeAndType& existing = (*to_update)[i];
    if (shapes_and_types[i].dtype == existing.dtype) {
      new_values[i].dtype = existing.dtype;
    } else {
      if (existing.dtype != DT_INVALID) {
        return false;
      } else {
        new_values[i].dtype = shapes_and_types[i].dtype;
        refined = true;
      }
    }
    if (!Merge(existing.shape, shapes_and_types[i].shape, &new_values[i].shape)
             .ok()) {
      // merge failed, ignore the new value.
      new_values[i].shape = existing.shape;
    }
    if (!existing.shape.SameHandle(new_values[i].shape)) {
      refined = true;
    }
  }
  if (!refined) {
    return false;
  }
  for (int i = 0; i < new_values.size(); ++i) {
    (*to_update)[i] = new_values[i];
  }
  return true;
}

bool InferenceContext::MergeOutputHandleShapesAndTypes(
    int idx, const std::vector<ShapeAndType>& shapes_and_types) {
  if (output_handle_shapes_and_types_[idx] == nullptr) {
    output_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
    return true;
  }
  return MergeHandleShapesAndTypes(shapes_and_types,
                                   output_handle_shapes_and_types_[idx].get());
}

bool InferenceContext::MergeInputHandleShapesAndTypes(
    int idx, const std::vector<ShapeAndType>& shapes_and_types) {
  if (input_handle_shapes_and_types_[idx] == nullptr) {
    input_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
    return true;
  }
  return MergeHandleShapesAndTypes(shapes_and_types,
                                   input_handle_shapes_and_types_[idx].get());
}

bool InferenceContext::RelaxHandleShapesAndMergeTypes(
    const std::vector<ShapeAndType>& shapes_and_types,
    std::vector<ShapeAndType>* to_update) {
  if (shapes_and_types.size() != to_update->size()) {
    return false;
  }
  std::vector<ShapeAndType> new_values(shapes_and_types.size());
  bool refined = false;
  for (int i = 0; i < shapes_and_types.size(); ++i) {
    const ShapeAndType& existing = (*to_update)[i];
    if (shapes_and_types[i].dtype == existing.dtype) {
      new_values[i].dtype = existing.dtype;
    } else {
      if (existing.dtype != DT_INVALID) {
        return false;
      } else {
        new_values[i].dtype = shapes_and_types[i].dtype;
        refined = true;
      }
    }
    Relax(existing.shape, shapes_and_types[i].shape, &new_values[i].shape);
    if (!existing.shape.SameHandle(new_values[i].shape)) {
      refined = true;
    }
  }
  if (!refined) {
    return false;
  }
  for (int i = 0; i < new_values.size(); ++i) {
    (*to_update)[i] = new_values[i];
  }
  return true;
}

bool InferenceContext::RelaxOutputHandleShapesAndMergeTypes(
    int idx, const std::vector<ShapeAndType>& shapes_and_types) {
  if (output_handle_shapes_and_types_[idx] == nullptr) {
    output_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
    return true;
  }
  return RelaxHandleShapesAndMergeTypes(
      shapes_and_types, output_handle_shapes_and_types_[idx].get());
}

bool InferenceContext::RelaxInputHandleShapesAndMergeTypes(
    int idx, const std::vector<ShapeAndType>& shapes_and_types) {
  if (input_handle_shapes_and_types_[idx] == nullptr) {
    input_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
    return true;
  }
  return RelaxHandleShapesAndMergeTypes(
      shapes_and_types, input_handle_shapes_and_types_[idx].get());
}

// -----------------------------------------------------------------------------
// ShapeManager
// -----------------------------------------------------------------------------
InferenceContext::ShapeManager::ShapeManager() {}
InferenceContext::ShapeManager::~ShapeManager() {
  for (auto* s : all_shapes_) delete s;
  for (auto* d : all_dims_) delete d;
}

ShapeHandle InferenceContext::ShapeManager::MakeShape(
    const std::vector<DimensionHandle>& dims) {
  all_shapes_.push_back(new Shape(dims));
  return all_shapes_.back();
}

ShapeHandle InferenceContext::ShapeManager::UnknownShape() {
  all_shapes_.push_back(new Shape());
  return all_shapes_.back();
}

}  // namespace shape_inference
}  // namespace tensorflow
