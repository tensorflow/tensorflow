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
#include "tensorflow/core/util/example_proto_helper.h"

#include <vector>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb_text.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

Status CheckValidType(const DataType& dtype) {
  switch (dtype) {
    case DT_INT64:
    case DT_FLOAT:
    case DT_STRING:
      return Status::OK();
    default:
      return errors::InvalidArgument("Received input dtype: ",
                                     DataTypeString(dtype));
  }
}

Status CheckTypesMatch(const Feature& feature, const DataType& dtype,
                       bool* match) {
  switch (dtype) {
    case DT_INT64:
      *match = (feature.kind_case() == Feature::kInt64List);
      break;
    case DT_FLOAT:
      *match = (feature.kind_case() == Feature::kFloatList);
      break;
    case DT_STRING:
      *match = (feature.kind_case() == Feature::kBytesList);
      break;
    default:
      return errors::InvalidArgument("Invalid input dtype: ",
                                     DataTypeString(dtype));
  }
  return Status::OK();
}

Status FeatureDenseCopy(const std::size_t out_index, const string& name,
                        const string& key, const DataType& dtype,
                        const TensorShape& shape, const Feature& feature,
                        Tensor* out) {
  const std::size_t num_elements = shape.num_elements();
  const std::size_t offset = out_index * num_elements;

  switch (dtype) {
    case DT_INT64: {
      const Int64List& values = feature.int64_list();
      if (static_cast<size_t>(values.value_size()) != num_elements) {
        return errors::InvalidArgument(
            "Name: ", name, ", Key: ", key, ", Index: ", out_index,
            ".  Number of int64 values != expected.  "
            "values size: ",
            values.value_size(), " but output shape: ", shape.DebugString());
      }
      auto out_p = out->flat<int64>().data() + offset;
      std::copy_n(values.value().data(), num_elements, out_p);
      return Status::OK();
    }
    case DT_FLOAT: {
      const FloatList& values = feature.float_list();
      if (static_cast<size_t>(values.value_size()) != num_elements) {
        return errors::InvalidArgument(
            "Name: ", name, ", Key: ", key, ", Index: ", out_index,
            ".  Number of float values != expected.  "
            "values size: ",
            values.value_size(), " but output shape: ", shape.DebugString());
      }
      auto out_p = out->flat<float>().data() + offset;
      std::copy_n(values.value().data(), num_elements, out_p);
      return Status::OK();
    }
    case DT_STRING: {
      const BytesList& values = feature.bytes_list();
      if (static_cast<size_t>(values.value_size()) != num_elements) {
        return errors::InvalidArgument(
            "Name: ", name, ", Key ", key, ", Index: ", out_index,
            ".  Number of bytes values != expected.  "
            "Values size: ",
            values.value_size(), " but output shape: ", shape.DebugString());
      }
      auto out_p = out->flat<string>().data() + offset;
      std::transform(values.value().data(),
                     values.value().data() + num_elements, out_p,
                     [](const string* s) { return *s; });
      return Status::OK();
    }
    default:
      return errors::InvalidArgument("Invalid input dtype: ",
                                     DataTypeString(dtype));
  }
}

Tensor FeatureSparseCopy(const std::size_t batch, const string& key,
                         const DataType& dtype, const Feature& feature) {
  switch (dtype) {
    case DT_INT64: {
      const Int64List& values = feature.int64_list();
      const int64 num_elements = values.value_size();
      Tensor out(dtype, TensorShape({num_elements}));
      auto out_p = out.flat<int64>().data();
      std::copy_n(values.value().data(), num_elements, out_p);
      return out;
    }
    case DT_FLOAT: {
      const FloatList& values = feature.float_list();
      const int64 num_elements = values.value_size();
      Tensor out(dtype, TensorShape({num_elements}));
      auto out_p = out.flat<float>().data();
      std::copy_n(values.value().data(), num_elements, out_p);
      return out;
    }
    case DT_STRING: {
      const BytesList& values = feature.bytes_list();
      const int64 num_elements = values.value_size();
      Tensor out(dtype, TensorShape({num_elements}));
      auto out_p = out.flat<string>().data();
      std::transform(values.value().data(),
                     values.value().data() + num_elements, out_p,
                     [](const string* s) { return *s; });
      return out;
    }
    default:
      LOG(FATAL) << "not supposed to be here.  dtype requested: " << dtype;
  }
}

int64 CopyIntoSparseTensor(const Tensor& in, const int batch,
                           const int64 offset, Tensor* indices,
                           Tensor* values) {
  const int64 num_elements = in.shape().num_elements();
  const DataType& dtype = in.dtype();
  CHECK_EQ(dtype, values->dtype());

  // Update indices.
  auto ix_t = indices->matrix<int64>();
  int64* ix_p = &ix_t(offset, 0);
  for (int64 i = 0; i < num_elements; ++i, ix_p += 2) {
    *ix_p = batch;    // Column 0 stores the batch entry
    *(ix_p + 1) = i;  // Column 1 stores the index in the batch
  }

  // Copy values over.
  switch (dtype) {
    case DT_INT64: {
      std::copy_n(in.flat<int64>().data(), num_elements,
                  values->flat<int64>().data() + offset);
      break;
    }
    case DT_FLOAT: {
      std::copy_n(in.flat<float>().data(), num_elements,
                  values->flat<float>().data() + offset);
      break;
    }
    case DT_STRING: {
      std::copy_n(in.flat<string>().data(), num_elements,
                  values->flat<string>().data() + offset);
      break;
    }
    default:
      LOG(FATAL) << "Not supposed to be here.  Saw dtype: " << dtype;
  }

  return num_elements;
}

void RowDenseCopy(const std::size_t& out_index, const DataType& dtype,
                  const Tensor& in, Tensor* out) {
  const std::size_t num_elements = in.shape().num_elements();
  const std::size_t offset = out_index * num_elements;

  switch (dtype) {
    case DT_INT64: {
      std::copy_n(in.flat<int64>().data(), num_elements,
                  out->flat<int64>().data() + offset);
      break;
    }
    case DT_FLOAT: {
      std::copy_n(in.flat<float>().data(), num_elements,
                  out->flat<float>().data() + offset);
      break;
    }
    case DT_STRING: {
      std::copy_n(in.flat<string>().data(), num_elements,
                  out->flat<string>().data() + offset);
      break;
    }
    default:
      LOG(FATAL) << "Not supposed to be here.  Saw dtype: " << dtype;
  }
}

Status SingleExampleProtoToTensors(
    const Example& example, const string& example_name, const int batch_index,
    const std::vector<FixedLenFeature>& fixed_len_features,
    const std::vector<VarLenFeature>& var_len_features,
    std::vector<Tensor*>* output_dense_values_tensor,
    std::vector<std::vector<Tensor>>* output_sparse_values_tmp) {
  const Features& features = example.features();
  const auto& feature_dict = features.feature();

  // Handle dense features.
  for (size_t d = 0; d < fixed_len_features.size(); ++d) {
    const FixedLenFeature& feature_config = fixed_len_features[d];
    const string& key = feature_config.key;
    const DataType& dtype = feature_config.dtype;
    const TensorShape& shape = feature_config.shape;
    const Tensor& default_value = feature_config.default_value;
    bool required = (default_value.NumElements() == 0);
    const auto& feature_found = feature_dict.find(key);
    const bool feature_has_data =  // Found key & data type is set
        (feature_found != feature_dict.end() &&
         (feature_found->second.kind_case() != Feature::KIND_NOT_SET));

    const bool required_ok = feature_has_data || !required;
    if (!required_ok) {
      return errors::InvalidArgument("Name: ", example_name, ", Feature: ", key,
                                     " is required but could not be found.");
    }

    // Perform the FeatureDenseCopy into the output dense_values tensor (if
    // the value is present).
    if (feature_has_data) {
      const Feature& f = feature_found->second;
      bool types_match;
      TF_RETURN_IF_ERROR(CheckTypesMatch(f, dtype, &types_match));
      if (!types_match) {
        return errors::InvalidArgument("Name: ", example_name, ", Feature: ",
                                       key, ".  Data types don't match. ",
                                       "Expected type: ", DataTypeString(dtype),
                                       "  Feature is: ", ProtoDebugString(f));
      }
      TF_RETURN_IF_ERROR(FeatureDenseCopy(batch_index, example_name, key, dtype,
                                          shape, f,
                                          (*output_dense_values_tensor)[d]));
    } else {
      // If the value is missing, RowDenseCopy the default value.
      RowDenseCopy(batch_index, dtype, default_value,
                   (*output_dense_values_tensor)[d]);
    }
  }

  // Handle sparse features.
  for (size_t d = 0; d < var_len_features.size(); ++d) {
    const VarLenFeature& feature_config = var_len_features[d];
    const string& key = feature_config.key;
    const DataType& dtype = feature_config.dtype;
    const auto& feature_found = feature_dict.find(key);

    const bool feature_has_data =  // Found key & data type is set
        (feature_found != feature_dict.end() &&
         (feature_found->second.kind_case() != Feature::KIND_NOT_SET));

    if (feature_has_data) {
      const Feature& f = feature_found->second;
      bool types_match;
      TF_RETURN_IF_ERROR(CheckTypesMatch(f, dtype, &types_match));
      if (!types_match) {
        return errors::InvalidArgument("Name: ", example_name, ", Feature: ",
                                       key, ".  Data types don't match. ",
                                       "Expected type: ", DataTypeString(dtype),
                                       "  Feature is: ", ProtoDebugString(f));
      }
      (*output_sparse_values_tmp)[d][batch_index] =
          FeatureSparseCopy(batch_index, key, dtype, f);
    } else {
      (*output_sparse_values_tmp)[d][batch_index] =
          Tensor(dtype, TensorShape({0}));
    }
  }
  return Status::OK();
}

Status GetSparseTensorShapes(const VarLenFeature& var_len_feature,
                             const std::vector<Tensor>& sparse_values_tmp,
                             const int batch_size,
                             VarLenFeatureBatchShapes* output_shapes) {
  int64 total_num_features = 0;
  int64 max_num_features = 0;
  for (int b = 0; b < batch_size; ++b) {
    const Tensor& t = sparse_values_tmp[b];
    const int64 num_elements = t.shape().num_elements();
    total_num_features += num_elements;
    max_num_features = std::max(max_num_features, num_elements);
  }
  output_shapes->indices_shape.AddDim(total_num_features);
  output_shapes->indices_shape.AddDim(2);
  output_shapes->values_shape.AddDim(total_num_features);
  output_shapes->max_num_features = max_num_features;
  return Status::OK();
}

Status BatchExampleProtoToTensors(
    const std::vector<const Example*>& examples,
    const std::vector<string>& names,
    const std::vector<FixedLenFeature>& fixed_len_features,
    const std::vector<VarLenFeature>& var_len_features, Allocator* allocator,
    std::vector<Tensor>* output_dense_values_tensor,
    std::vector<Tensor>* output_sparse_indices_tensor,
    std::vector<Tensor>* output_sparse_values_tensor,
    std::vector<Tensor>* output_sparse_shapes_tensor) {
  const int batch_size = examples.size();

  const bool has_names = (!names.empty());
  if (has_names) {
    if (names.size() != examples.size()) {
      return errors::InvalidArgument(
          "Expected len(names) == len(examples), but got: ", names.size(),
          " vs. ", examples.size());
    }
  }

  // We also need a map of Tensor pointers for the SingleExampleProtoToTensors
  // call. (Is there a better solution here?)
  std::vector<Tensor*> output_dense_values_tensor_ptrs(
      fixed_len_features.size());

  // Preallocate dense_values, since we know their sizes.
  for (size_t d = 0; d < fixed_len_features.size(); ++d) {
    const FixedLenFeature& config = fixed_len_features[d];
    TensorShape out_shape;
    out_shape.AddDim(batch_size);
    const TensorShape& shape = config.shape;
    const DataType& dtype = config.dtype;
    for (const int dim : shape.dim_sizes()) out_shape.AddDim(dim);
    (*output_dense_values_tensor)[d] = Tensor(allocator, dtype, out_shape);
    output_dense_values_tensor_ptrs[d] = &(*output_dense_values_tensor)[d];
  }

  // Temporary vector to hold sparse values.
  std::vector<std::vector<Tensor>> sparse_values_tmp(var_len_features.size());

  for (size_t d = 0; d < var_len_features.size(); ++d) {
    sparse_values_tmp[d] = std::vector<Tensor>(batch_size);
  }

  for (size_t b = 0; b < examples.size(); ++b) {
    const Example& ex = *(examples[b]);
    const string& example_name = (has_names) ? names[b] : "<unknown>";
    TF_RETURN_IF_ERROR(SingleExampleProtoToTensors(
        ex, example_name, b, fixed_len_features, var_len_features,
        &output_dense_values_tensor_ptrs, &sparse_values_tmp));
  }

  for (size_t d = 0; d < var_len_features.size(); ++d) {
    const VarLenFeature& feature_config = var_len_features[d];
    const DataType& dtype = feature_config.dtype;
    const std::vector<Tensor>& sparse_values_tensor = sparse_values_tmp[d];

    VarLenFeatureBatchShapes sparse_tensor_batch_shapes;
    TF_RETURN_IF_ERROR(GetSparseTensorShapes(feature_config,
                                             sparse_values_tensor, batch_size,
                                             &sparse_tensor_batch_shapes));
    const TensorShape& indices_shape = sparse_tensor_batch_shapes.indices_shape;
    const TensorShape& values_shape = sparse_tensor_batch_shapes.values_shape;

    // Allocate the sparse indices here.
    (*output_sparse_indices_tensor)[d] =
        Tensor(allocator, DT_INT64, indices_shape);
    (*output_sparse_values_tensor)[d] = Tensor(allocator, dtype, values_shape);
    (*output_sparse_shapes_tensor)[d] =
        Tensor(allocator, DT_INT64, TensorShape({2}));

    auto shape_t = (*output_sparse_shapes_tensor)[d].vec<int64>();
    shape_t(0) = batch_size;
    shape_t(1) = sparse_tensor_batch_shapes.max_num_features;

    Tensor* sp_indices_d = &(*output_sparse_indices_tensor)[d];
    Tensor* sp_values_d = &(*output_sparse_values_tensor)[d];

    int64 offset = 0;
    for (int b = 0; b < batch_size; ++b) {
      const int64 num_elements = CopyIntoSparseTensor(
          sparse_values_tensor[b], b, offset, sp_indices_d, sp_values_d);
      offset += num_elements;
    }
  }
  return Status::OK();
}

Status ParseExampleAttrs::FinishInit() {
  if (static_cast<size_t>(num_sparse) != sparse_types.size()) {
    return errors::InvalidArgument("len(sparse_keys) != len(sparse_types)");
  }
  if (static_cast<size_t>(num_dense) != dense_types.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_types)");
  }
  if (static_cast<size_t>(num_dense) != dense_shapes.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_shapes)");
  }
  if (num_dense > std::numeric_limits<int32>::max()) {
    return errors::InvalidArgument("num_dense_ too large");
  }
  for (const DataType& type : dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  return Status::OK();
}

Status ParseSingleExampleAttrs::FinishInit() {
  if (sparse_keys.size() != sparse_types.size()) {
    return errors::InvalidArgument("len(sparse_keys) != len(sparse_types)");
  }
  if (dense_keys.size() != dense_types.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_types)");
  }
  if (dense_keys.size() != dense_shapes.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_shapes)");
  }
  for (const DataType& type : dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  return Status::OK();
}

Status ParseSingleSequenceExampleAttrs::FinishInit() {
  if (static_cast<size_t>(num_context_sparse) != context_sparse_types.size()) {
    return errors::InvalidArgument(
        "len(context_sparse_keys) != len(context_sparse_types)");
  }
  if (static_cast<size_t>(num_context_dense) != context_dense_types.size()) {
    return errors::InvalidArgument(
        "len(context_dense_keys) != len(context_dense_types)");
  }
  if (static_cast<size_t>(num_context_dense) != context_dense_shapes.size()) {
    return errors::InvalidArgument(
        "len(context_dense_keys) != len(context_dense_shapes)");
  }
  if (static_cast<size_t>(num_feature_list_sparse) !=
      feature_list_sparse_types.size()) {
    return errors::InvalidArgument(
        "len(feature_list_sparse_keys) != len(feature_list_sparse_types)");
  }
  if (static_cast<size_t>(num_feature_list_dense) !=
      feature_list_dense_types.size()) {
    return errors::InvalidArgument(
        "len(feature_list_dense_keys) != "
        "len(feature_list_dense_types)");
  }
  for (const DataType& type : context_dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : context_sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : feature_list_dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : feature_list_sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  return Status::OK();
}

}  // namespace tensorflow
