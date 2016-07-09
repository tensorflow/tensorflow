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
#include "tensorflow/core/framework/shape_inference_testutil.h"

#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

using shape_inference::Dimension;
using shape_inference::Shape;
using errors::Unknown;

Status InferShapes(const string& op_name, const string& ins,
                   const string& expected_outs, const NodeDef* node_def,
                   const std::vector<const Tensor*>& input_tensors) {
  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUp(op_name, &op_reg_data));
  const int num_outputs = op_reg_data->op_def.output_arg_size();

  std::vector<string> ins_v = str_util::Split(ins, ';');
  std::unique_ptr<const NodeDef> new_node_def;
  if (node_def == nullptr) {
    new_node_def.reset(new NodeDef);
    node_def = new_node_def.get();
  }
  shape_inference::InferenceContext c(node_def, ins_v, num_outputs,
                                      input_tensors);
  TF_RETURN_IF_ERROR(op_reg_data->shape_inference_fn(&c));

  std::unordered_map<const Dimension*, std::pair<int, int>>
      dim_to_input_and_dim_idx;
  std::unordered_map<const Shape*, int> shape_to_input_idx;
  for (int i = 0; i < c.num_inputs(); ++i) {
    auto in = c.input(i);
    shape_to_input_idx[in] = i;
    for (int j = 0; j < c.Rank(in); ++j) {
      dim_to_input_and_dim_idx[c.Dim(in, j)] = std::make_pair(i, j);
    }
  }
  if (expected_outs == "e") {
    return Unknown("Shape inference should have returned error");
  }

  // Verify the output shape.
  std::vector<string> expected_outs_v = str_util::Split(expected_outs, ';');
  if (num_outputs != expected_outs_v.size()) {
    return Unknown("Wrong number of expected outputs (", expected_outs_v.size(),
                   " vs ", num_outputs, ")");
  }
  for (int i = 0; i < num_outputs; ++i) {
    string err_prefix = strings::StrCat("Output ", i);
    StringPiece expected(expected_outs_v[i]);
    const shape_inference::Shape* out = c.output(i);
    const int in_index = gtl::FindWithDefault(shape_to_input_idx, out, -1);
    if (expected.starts_with("in")) {
      if (in_index == -1) {
        return Unknown(err_prefix, " did not match any input shape");
      }
      auto v = str_util::Split(expected, '|');
      if (std::find(v.begin(), v.end(), strings::StrCat("in", in_index)) ==
          v.end()) {
        return Unknown(err_prefix, " matched input ", in_index,
                       " and should have matched one of (", expected, ")");
      }
      continue;
    }
    if (in_index != -1) {
      return Unknown(err_prefix, " matched input ", in_index,
                     " and should have not matched an input shape");
    }
    if (expected == "?") {
      if (c.RankKnown(out)) {
        return Unknown(err_prefix, " expected to be unknown but was ",
                       c.DebugString(out));
      }
      continue;
    }

    // Verify the dimensions.
    CHECK(expected.starts_with("[") && expected.ends_with("]"));
    expected.remove_prefix(1);
    expected.remove_suffix(1);

    // Split expected as a dimension.
    auto expected_dims = str_util::Split(expected, ',');
    if (!c.RankKnown(out)) {
      return Unknown(err_prefix, " expected rank ", expected_dims.size(),
                     " but was ?");
    }
    if (c.Rank(out) != expected_dims.size()) {
      return Unknown(err_prefix, " expected rank ", expected_dims.size(),
                     " but was ", c.Rank(out));
    }
    for (int j = 0; j < expected_dims.size(); ++j) {
      err_prefix = strings::StrCat("Output dim ", i, ",", j);
      StringPiece expected_dim(expected_dims[j]);
      const Dimension* out_dim = c.Dim(out, j);
      std::pair<int, int> in_dim_idx = gtl::FindWithDefault(
          dim_to_input_and_dim_idx, out_dim, std::make_pair(-1, -1));
      if (expected_dim == "?") {
        if (in_dim_idx.first != -1) {
          return Unknown(err_prefix,
                         " expected to be unknown but matched input d",
                         in_dim_idx.first, "_", in_dim_idx.second);
        } else if (c.ValueKnown(out_dim)) {
          return Unknown(err_prefix, " expected to be unknown but was ",
                         c.Value(out_dim));
        }
      } else if (expected_dim.starts_with("d")) {
        // Compare the dimension values.
        auto v = str_util::Split(expected_dim, '|');
        if (in_dim_idx.first == -1) {
          return Unknown(err_prefix, " did not match any input dim");
        }
        if (std::find(v.begin(), v.end(),
                      strings::StrCat("d", in_dim_idx.first, "_",
                                      in_dim_idx.second)) == v.end()) {
          return Unknown(err_prefix, " matched input d", in_dim_idx.first, "_",
                         in_dim_idx.second, " and should have matched one of ",
                         expected_dim);
        }
      } else {
        // Parse it as a value.
        int64 value = -1;
        if (!strings::safe_strto64(expected_dim, &value)) {
          return Unknown(err_prefix, " expected dim failed to parse as int64");
        }
        if (in_dim_idx.first != -1) {
          return Unknown(err_prefix, " expected to be ", value,
                         " but matched input d", in_dim_idx.first, "_",
                         in_dim_idx.second);
        } else if (value != c.Value(out_dim)) {
          return Unknown(err_prefix, " expected to be ", value, " but was ",
                         c.DebugString(out_dim));
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
