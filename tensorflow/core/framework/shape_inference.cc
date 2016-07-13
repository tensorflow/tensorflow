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

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace shape_inference {

constexpr int32 InferenceContext::kUnknownRank;
constexpr int64 InferenceContext::kUnknownDim;

InferenceContext::InferenceContext(
    const NodeDef* node_def, const std::vector<string>& input_shapes,
    int num_outputs, const std::vector<const Tensor*>& input_tensors)
    : input_tensors_(input_tensors), node_def_(*CHECK_NOTNULL(node_def)) {
  for (const string& spec : input_shapes) {
    if (spec == "?") {
      inputs_.push_back(CreateUnknownShape());
    } else {
      std::vector<const Dimension*> dims;
      strings::Scanner scanner(spec);
      scanner.OneLiteral("[");
      while (scanner.Peek() != ']') {
        if (scanner.Peek() == '?') {
          scanner.OneLiteral("?");
          dims.push_back(CreateUnknownDim());
        } else {
          scanner.RestartCapture().Many(strings::Scanner::DIGIT);
          StringPiece match;
          int64 dim_size = 0;
          CHECK(scanner.GetResult(nullptr, &match) &&
                strings::safe_strto64(match, &dim_size))
              << spec;
          dims.push_back(CreateDim(dim_size));
        }

        if (scanner.Peek() == ',') {
          scanner.OneLiteral(",");
        } else {
          CHECK_EQ(scanner.Peek(), ']');
        }
      }
      CHECK(scanner.OneLiteral("]").Eos().GetResult()) << spec;
      inputs_.push_back(CreateShape(dims));
    }
  }

  CHECK_LE(input_tensors_.size(), input_shapes.size());
  input_tensors_.resize(input_shapes.size());

  for (int i = 0; i < num_outputs; ++i) {
    outputs_.push_back(CreateUnknownShape());
  }
}

InferenceContext::~InferenceContext() {
  for (auto* s : all_shapes_) delete s;
  for (auto* d : all_dims_) delete d;
}

string InferenceContext::DebugString(const Shape* s) {
  if (RankKnown(s)) {
    std::vector<string> vals;
    for (auto d : s->dims_) vals.push_back(DebugString(d));
    return strings::StrCat("[", str_util::Join(vals, ","), "]");
  } else {
    return "?";
  }
}

string InferenceContext::DebugString(const Dimension* d) {
  return ValueKnown(d) ? strings::StrCat(Value(d)) : "?";
}

Status InferenceContext::WithRank(const Shape* shape, int32 rank,
                                  const Shape** out) {
  const int32 existing = Rank(shape);
  if (existing == rank) {
    *out = shape;
    return Status::OK();
  }
  if (existing == kUnknownRank) {
    std::vector<const Dimension*> dims;
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

Status InferenceContext::WithRankAtLeast(const Shape* shape, int32 rank,
                                         const Shape** out) {
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

Status InferenceContext::WithRankAtMost(const Shape* shape, int32 rank,
                                        const Shape** out) {
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

Status InferenceContext::WithValue(const Dimension* dim, int64 value,
                                   const Dimension** out) {
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

Status InferenceContext::Merge(const Dimension* d0, const Dimension* d1,
                               const Dimension** out) {
  if (d0 == d1 || !ValueKnown(d1)) {
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

Status InferenceContext::MergePrefix(const Shape* s, const Shape* prefix,
                                     const Shape** s_out,
                                     const Shape** prefix_out) {
  *s_out = *prefix_out = nullptr;
  if (!RankKnown(prefix) || !RankKnown(s)) {
    *s_out = s;
    *prefix_out = prefix;
    return Status::OK();
  }
  const int32 rank = Rank(prefix);
  TF_RETURN_IF_ERROR(WithRankAtLeast(s, rank, &s));

  // Merge the prefix dims and create the new output shapes.
  std::vector<const Dimension*> dims;
  dims.resize(rank);
  for (int i = 0; i < rank; ++i) {
    TF_RETURN_IF_ERROR(Merge(Dim(s, i), Dim(prefix, i), &dims[i]));
  }
  *prefix_out = CreateShape(dims);
  for (int i = rank; i < Rank(s); ++i) dims.push_back(Dim(s, i));
  *s_out = CreateShape(dims);
  return Status::OK();
}

Status InferenceContext::Merge(const Shape* s0, const Shape* s1,
                               const Shape** out) {
  if (s0 == s1 || !RankKnown(s1)) {
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
    if (d0 == d1) continue;

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
  std::vector<const Dimension*> dims(rank, nullptr);
  for (int i = 0; i < rank; ++i) {
    // Invariant for merge was checked earlier, so CHECK is ok.
    TF_CHECK_OK(Merge(Dim(s0, i), Dim(s1, i), &dims[i]));
  }
  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::Subshape(const Shape* s, int64 start,
                                  const Shape** out) {
  return Subshape(s, start, std::numeric_limits<int64>::max() /* end */, out);
}

Status InferenceContext::Subshape(const Shape* s, int64 start_in, int64 end_in,
                                  const Shape** out) {
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
  std::vector<const Dimension*> dims;
  dims.reserve(end - start);
  for (int i = start; i < end; ++i) {
    dims.push_back(Dim(s, i));
  }
  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::Concatenate(const Shape* s1, const Shape* s2,
                                     const Shape** out) {
  if (!RankKnown(s1) || !RankKnown(s2)) {
    return ReturnUnknownShape(out);
  }
  const int32 s1_rank = Rank(s1);
  const int32 s2_rank = Rank(s2);
  const int32 rank = s1_rank + s2_rank;
  std::vector<const Dimension*> dims;
  dims.reserve(rank);
  for (int i = 0; i < s1_rank; ++i) dims.push_back(Dim(s1, i));
  for (int i = 0; i < s2_rank; ++i) dims.push_back(Dim(s2, i));
  return ReturnCreatedShape(dims, out);
}

const Shape* InferenceContext::CreateShape(
    const std::vector<const Dimension*>& dims) {
  all_shapes_.push_back(new Shape(dims));
  return all_shapes_.back();
}

const Shape* InferenceContext::CreateUnknownShape() {
  all_shapes_.push_back(new Shape());
  return all_shapes_.back();
}

Status InferenceContext::CreateShapeFromShapeTensor(int input_idx,
                                                    const Shape** out) {
  const Shape* input_shape;
  TF_RETURN_IF_ERROR(WithRank(input(input_idx), 1, &input_shape));

  const Tensor* t = input_tensor(input_idx);
  if (t == nullptr) {
    return ReturnUnknownShape(out);
  }
  if (t->shape().dims() != 1) {
    *out = nullptr;
    return errors::InvalidArgument("Input tensor must be rank 1, but was rank ",
                                   t->shape().dims());
  }
  std::vector<const Dimension*> dims;
  if (t->dtype() == DataType::DT_INT32) {
    auto flat_t = t->flat<int32>();
    for (int i = 0; i < flat_t.size(); ++i) {
      dims.push_back(CreateDim(flat_t(i)));
    }
  } else if (t->dtype() == DataType::DT_INT64) {
    auto flat_t = t->flat<int64>();
    for (int i = 0; i < flat_t.size(); ++i) {
      dims.push_back(CreateDim(flat_t(i)));
    }
  } else {
    *out = nullptr;
    return errors::InvalidArgument(
        "Input tensor must be int32 or int64, but was ",
        DataTypeString(t->dtype()));
  }

  return ReturnCreatedShape(dims, out);
}

const Dimension* InferenceContext::CreateDim(int64 value) {
  all_dims_.push_back(new Dimension(value));
  return all_dims_.back();
}

// Returns a new dimension whose value is given by a scalar input tensor.
Status InferenceContext::CreateDimForScalarInput(int idx,
                                                 const Dimension** out) {
  const Tensor* t = input_tensor(idx);
  if (t == nullptr) {
    *out = CreateUnknownDim();
    return Status::OK();
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
  *out = CreateDim(val);
  return Status::OK();
}

const Dimension* InferenceContext::CreateUnknownDim() {
  all_dims_.push_back(new Dimension());
  return all_dims_.back();
}

}  // namespace shape_inference
}  // namespace tensorflow
