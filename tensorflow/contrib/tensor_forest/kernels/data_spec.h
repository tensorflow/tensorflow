// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
// This is a surrogate for using a proto, since it doesn't seem to be possible
// to use protos in a dynamically-loaded/shared-linkage library, which is
// what is used for custom ops in tensorflow/contrib.
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_CORE_OPS_DATA_SPEC_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_CORE_OPS_DATA_SPEC_H_
#include <unordered_map>

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace tensorforest {

using tensorflow::strings::safe_strto32;

// DataColumn holds information about one feature of the original data.
// A feature could be dense or sparse, and be of any size.
class DataColumn {
 public:
  DataColumn() {}

  // Parses a serialized DataColumn produced from the SerializeToString()
  // function of a python data_ops.DataColumn object.
  // It should look like a proto ASCII format, i.e.
  // name: <name> original_type: <type> size: <size>
  void ParseFromString(const string& serialized) {
    std::vector<string> tokens = tensorflow::str_util::Split(serialized, ' ');
    CHECK_EQ(tokens.size(), 6);
    name_ = tokens[1];
    safe_strto32(tokens[3], &original_type_);
    safe_strto32(tokens[5], &size_);
  }

  const string& name() const { return name_; }

  int original_type() const { return original_type_; }

  int size() const { return size_; }

  void set_name(const string& n) { name_ = n; }

  void set_original_type(int o) { original_type_ = o; }

  void set_size(int s) { size_ = s; }

 private:
  string name_;
  int original_type_;
  int size_;
};

// TensorForestDataSpec holds information about the original features of the
// data set, which were flattened to a single dense float tensor and/or a
// single sparse float tensor.
class TensorForestDataSpec {
 public:
  TensorForestDataSpec() {}

  // Parses a serialized DataColumn produced from the SerializeToString()
  // function of a python data_ops.TensorForestDataSpec object.
  // It should look something like:
  // dense_features_size: <size> dense: [{<col1>}{<col2>}] sparse: [{<col3>}]
  void ParseFromString(const string& serialized) {
    std::vector<string> tokens = tensorflow::str_util::Split(serialized, "[]");
    std::vector<string> first_part =
        tensorflow::str_util::Split(tokens[0], ' ');
    safe_strto32(first_part[1], &dense_features_size_);
    ParseColumns(tokens[1], &dense_);
    ParseColumns(tokens[3], &sparse_);

    int total = 0;
    for (const DataColumn& col : dense_) {
      for (int i = 0; i < col.size(); ++i) {
        feature_to_type_.push_back(col.original_type());
        ++total;
      }
    }
  }

  const DataColumn& dense(int i) const { return dense_.at(i); }

  const DataColumn& sparse(int i) const { return sparse_.at(i); }

  DataColumn* mutable_sparse(int i) { return &sparse_[i]; }

  int dense_size() const { return dense_.size(); }

  int sparse_size() const { return sparse_.size(); }

  int dense_features_size() const { return dense_features_size_; }

  void set_dense_features_size(int s) { dense_features_size_ = s; }

  DataColumn* add_dense() {
    dense_.push_back(DataColumn());
    return &dense_[dense_.size() - 1];
  }

  int GetDenseFeatureType(int feature) const {
    return feature_to_type_[feature];
  }

 private:
  void ParseColumns(const string& cols, std::vector<DataColumn>* vec) {
    std::vector<string> tokens = tensorflow::str_util::Split(cols, "{}");
    for (const string& tok : tokens) {
      if (!tok.empty()) {
        DataColumn col;
        col.ParseFromString(tok);
        vec->push_back(col);
      }
    }
  }

  std::vector<DataColumn> dense_;
  std::vector<DataColumn> sparse_;
  int dense_features_size_;

  // This map tracks features in the total dense feature space to their
  // original type for fast lookup.
  std::vector<int> feature_to_type_;
};

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_CORE_OPS_DATA_SPEC_H_
