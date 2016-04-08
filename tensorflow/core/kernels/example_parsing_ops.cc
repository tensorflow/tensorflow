/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/parsing_ops.cc.

#include <unordered_set>

#include <vector>
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

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
      CHECK(false) << "not supposed to be here.  dtype requested: " << dtype;
  }
}

int64 CopyIntoSparseTensor(const Tensor& in, const int batch,
                           const int64 offset, Tensor* indices,
                           Tensor* values) {
  const int64 num_elements = in.shape().num_elements();
  const DataType& dtype = in.dtype();
  CHECK_EQ(dtype, values->dtype());

  // Update indices
  auto ix_t = indices->matrix<int64>();
  int64* ix_p = &ix_t(offset, 0);
  for (int64 i = 0; i < num_elements; ++i, ix_p += 2) {
    *ix_p = batch;    // Column 0 stores the batch entry
    *(ix_p + 1) = i;  // Column 1 stores the index in the batch
  }

  // Copy values over
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
      // auto values_t = values->flat<string>().data() + offset;
      // auto in_t = in.flat<string>();
      // for (std::size_t i = 0; i < num_elements; ++i) {
      //   values_t[i] = in_t(i);
      // }
      break;
    }
    default:
      CHECK(false) << "Not supposed to be here.  Saw dtype: " << dtype;
  }

  return num_elements;
}

void RowDenseCopy(const std::size_t& batch, const DataType& dtype,
                  const Tensor& in, Tensor* out) {
  const std::size_t num_elements = in.shape().num_elements();
  const std::size_t offset = batch * num_elements;

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
      CHECK(false) << "Not supposed to be here.  Saw dtype: " << dtype;
  }
}

}  // namespace

class ExampleParserOp : public OpKernel {
 public:
  explicit ExampleParserOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_types", &sparse_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Ndense", &num_dense_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Nsparse", &num_sparse_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tdense", &dense_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_shapes", &dense_shapes_));

    OP_REQUIRES(
        ctx, static_cast<size_t>(num_sparse_) == sparse_types_.size(),
        errors::InvalidArgument("len(sparse_keys) != len(sparse_types"));
    OP_REQUIRES(ctx, static_cast<size_t>(num_dense_) == dense_types_.size(),
                errors::InvalidArgument("len(dense_keys) != len(dense_types"));
    OP_REQUIRES(ctx, static_cast<size_t>(num_dense_) == dense_shapes_.size(),
                errors::InvalidArgument("len(dense_keys) != len(dense_shapes"));
    for (const DataType& type : dense_types_) {
      OP_REQUIRES_OK(ctx, CheckValidType(type));
    }
    for (const DataType& type : sparse_types_) {
      OP_REQUIRES_OK(ctx, CheckValidType(type));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* names;
    const Tensor* serialized;
    OpInputList dense_keys;
    OpInputList sparse_keys;
    OpInputList dense_defaults;

    OP_REQUIRES_OK(ctx, ctx->input("names", &names));
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_keys", &dense_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("sparse_keys", &sparse_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_defaults", &dense_defaults));

    std::vector<string> dense_keys_t(num_dense_);
    std::vector<string> sparse_keys_t(num_sparse_);
    CHECK_EQ(dense_keys.size(), num_dense_);
    CHECK_EQ(sparse_keys.size(), num_sparse_);
    for (int di = 0; di < num_dense_; ++di) {
      dense_keys_t[di] = dense_keys[di].scalar<string>()();
    }
    for (int di = 0; di < num_sparse_; ++di) {
      sparse_keys_t[di] = sparse_keys[di].scalar<string>()();
    }

    bool has_names = (names->NumElements() > 0);
    if (has_names) {
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsVector(names->shape()),
          errors::InvalidArgument("Expected names to be a vector, got shape: ",
                                  names->shape().DebugString()));
      OP_REQUIRES(
          ctx, names->NumElements() == serialized->NumElements(),
          errors::InvalidArgument(
              "Expected len(names) == len(serialized), but got: ",
              names->NumElements(), " vs. ", serialized->NumElements()));
    }
    auto names_t = names->flat<string>();

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(serialized->shape()),
                errors::InvalidArgument(
                    "Expected serialized to be a vector, got shape: ",
                    serialized->shape().DebugString()));
    OP_REQUIRES(ctx, dense_defaults.size() == num_dense_,
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_defaults.size(), " vs. ", num_dense_));

    std::vector<bool> required(num_dense_);
    for (int d = 0; d < num_dense_; ++d) {
      const Tensor& def_value = dense_defaults[d];
      required[d] = (def_value.NumElements() == 0);  // No default provided.

      if (def_value.NumElements() > 0) {
        OP_REQUIRES(ctx, def_value.shape() == dense_shapes_[d],
                    errors::InvalidArgument("def_value[", d, "].shape() == ",
                                            def_value.shape().DebugString(),
                                            " != dense_shapes_[", d, "] == ",
                                            dense_shapes_[d].DebugString()));
        OP_REQUIRES(ctx, def_value.dtype() == dense_types_[d],
                    errors::InvalidArgument(
                        "dense_defaults[", d, "].dtype() == ",
                        DataTypeString(def_value.dtype()), " != dense_types_[",
                        d, "] == ", DataTypeString(dense_types_[d])));
      }
    }

    auto serialized_t = serialized->vec<string>();

    const int batch_size = serialized_t.size();

    OpOutputList sparse_indices;
    OpOutputList sparse_values;
    OpOutputList sparse_shapes;
    OpOutputList dense_values;

    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_indices", &sparse_indices));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_values", &sparse_values));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_shapes", &sparse_shapes));
    OP_REQUIRES_OK(ctx, ctx->output_list("dense_values", &dense_values));

    // Preallocate dense_values, since we know their sizes
    for (int d = 0; d < num_dense_; ++d) {
      TensorShape out_shape;
      out_shape.AddDim(batch_size);
      for (const int dim : dense_shapes_[d].dim_sizes()) out_shape.AddDim(dim);
      Tensor* out = nullptr;
      dense_values.allocate(d, out_shape, &out);
    }

    // sparse_values_tmp will be num_sparse_ x batch_size, containing
    // the sparse values from the input layer.  after these are all
    // stored, we can allocate properly sized outputs and copy data over.
    // Doing it this way saves us the trouble of either performing
    // deserialization twice, or alternatively storing all copies of
    // the full Example protos.
    std::vector<std::vector<Tensor> > sparse_values_tmp(num_sparse_);

    Example ex;
    for (std::size_t b = 0; b < static_cast<size_t>(batch_size); ++b) {
      OP_REQUIRES(
          ctx, ParseProtoUnlimited(&ex, serialized_t(b)),
          errors::InvalidArgument("Could not parse example input, value: '",
                                  serialized_t(b), "'"));

      const string& name = (has_names) ? names_t(b) : "<unknown>";
      const Features& features = ex.features();
      const auto& feature_dict = features.feature();

      // Dense -----------------------------------------------------------------
      for (int d = 0; d < num_dense_; ++d) {
        const string& key = dense_keys_t[d];
        const DataType& dtype = dense_types_[d];
        const TensorShape& shape = dense_shapes_[d];

        const auto& feature_found = feature_dict.find(key);
        OP_REQUIRES(
            ctx, (feature_found != feature_dict.end()) || !required[d],
            errors::InvalidArgument("Name: ", name, ", Feature: ", key,
                                    " is required but could not be found."));
        if (feature_found != feature_dict.end()) {
          const Feature& f = feature_found->second;
          bool types_match;
          OP_REQUIRES_OK(ctx, CheckTypesMatch(f, dtype, &types_match));
          OP_REQUIRES(
              ctx, types_match,
              errors::InvalidArgument("Name: ", name, ", Feature: ", key,
                                      ".  Data types don't match. ",
                                      "Expected type: ", DataTypeString(dtype),
                                      "  Feature is: ", f.DebugString()));

          OP_REQUIRES_OK(ctx, FeatureDenseCopy(b, name, key, dtype, shape, f,
                                               dense_values[d]));
        } else {
          RowDenseCopy(b, dtype, dense_defaults[d], dense_values[d]);
        }
      }

      // Sparse ----------------------------------------------------------------
      for (int d = 0; d < num_sparse_; ++d) {
        const string& key = sparse_keys_t[d];
        const DataType& dtype = sparse_types_[d];

        const auto& feature_found = feature_dict.find(key);
        bool feature_has_data =  // Found key & data type is set
            (feature_found != feature_dict.end() &&
             (feature_found->second.kind_case() != Feature::KIND_NOT_SET));
        if (feature_has_data) {
          const Feature& f = feature_found->second;
          bool types_match;
          OP_REQUIRES_OK(ctx, CheckTypesMatch(f, dtype, &types_match));
          OP_REQUIRES(
              ctx, types_match,
              errors::InvalidArgument("Name: ", name, ", Feature: ", key,
                                      ".  Data types don't match. ",
                                      "Expected type: ", DataTypeString(dtype),
                                      "  Feature is: ", f.DebugString()));
          sparse_values_tmp[d].push_back(FeatureSparseCopy(b, key, dtype, f));
        } else {
          sparse_values_tmp[d].push_back(Tensor(dtype, TensorShape({0})));
        }
      }
    }

    // Copy sparse data into its final resting Tensors -------------------------
    for (int d = 0; d < num_sparse_; ++d) {
      int64 total_num_features = 0;
      int64 max_num_features = 0;
      for (int b = 0; b < batch_size; ++b) {
        const Tensor& t = sparse_values_tmp[d][b];
        const int64 num_elements = t.shape().num_elements();
        total_num_features += num_elements;
        max_num_features = std::max(max_num_features, num_elements);
      }

      TensorShape indices_shape({total_num_features, 2});
      TensorShape values_shape({total_num_features});
      Tensor* sp_indices_d = nullptr;
      Tensor* sp_values_d = nullptr;
      Tensor* sp_shape_d = nullptr;
      sparse_indices.allocate(d, indices_shape, &sp_indices_d);
      sparse_values.allocate(d, values_shape, &sp_values_d);
      sparse_shapes.allocate(d, TensorShape({2}), &sp_shape_d);

      auto shape_t = sp_shape_d->vec<int64>();
      shape_t(0) = batch_size;
      shape_t(1) = max_num_features;

      int64 offset = 0;

      for (int b = 0; b < batch_size; ++b) {
        const int64 num_elements = CopyIntoSparseTensor(
            sparse_values_tmp[d][b], b, offset, sp_indices_d, sp_values_d);
        offset += num_elements;
      }
    }
  }

 protected:
  int64 num_sparse_;
  int64 num_dense_;
  std::vector<DataType> sparse_types_;
  std::vector<DataType> dense_types_;
  std::vector<TensorShape> dense_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("ParseExample").Device(DEVICE_CPU),
                        ExampleParserOp);

class SingleSequenceExampleParserOp : public OpKernel {
 public:
  explicit SingleSequenceExampleParserOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("context_sparse_types", &context_sparse_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Ncontext_dense", &num_context_dense_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("Nfeature_list_dense", &num_feature_list_dense_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Ncontext_sparse", &num_context_sparse_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tcontext_dense", &context_dense_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_list_sparse_types",
                                     &feature_list_sparse_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_list_dense_types",
                                     &feature_list_dense_types_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("Nfeature_list_sparse", &num_feature_list_sparse_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("context_dense_shapes", &context_dense_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_list_dense_shapes",
                                     &feature_list_dense_shapes_));

    OP_REQUIRES(ctx, static_cast<size_t>(num_context_sparse_) ==
                         context_sparse_types_.size(),
                errors::InvalidArgument(
                    "len(context_sparse_keys) != len(context_sparse_types"));
    OP_REQUIRES(ctx, static_cast<size_t>(num_context_dense_) ==
                         context_dense_types_.size(),
                errors::InvalidArgument(
                    "len(context_dense_keys) != len(context_dense_types"));
    OP_REQUIRES(ctx, static_cast<size_t>(num_context_dense_) ==
                         context_dense_shapes_.size(),
                errors::InvalidArgument(
                    "len(context_dense_keys) != len(context_dense_shapes"));
    OP_REQUIRES(
        ctx, static_cast<size_t>(num_feature_list_sparse_) ==
                 feature_list_sparse_types_.size(),
        errors::InvalidArgument(
            "len(feature_list_sparse_keys) != len(feature_list_sparse_types"));
    OP_REQUIRES(ctx, static_cast<size_t>(num_feature_list_dense_) ==
                         feature_list_dense_types_.size(),
                errors::InvalidArgument("len(feature_list_dense_keys) != "
                                        "len(feature_list_dense_types"));
    for (const DataType& type : context_dense_types_) {
      OP_REQUIRES_OK(ctx, CheckValidType(type));
    }
    for (const DataType& type : context_sparse_types_) {
      OP_REQUIRES_OK(ctx, CheckValidType(type));
    }
    for (const DataType& type : feature_list_dense_types_) {
      OP_REQUIRES_OK(ctx, CheckValidType(type));
    }
    for (const DataType& type : feature_list_sparse_types_) {
      OP_REQUIRES_OK(ctx, CheckValidType(type));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* debug_name;
    const Tensor* serialized;
    OpInputList context_dense_keys;
    OpInputList context_sparse_keys;
    OpInputList context_dense_defaults;
    OpInputList feature_list_dense_keys;
    OpInputList feature_list_sparse_keys;
    const Tensor* feature_list_dense_missing_assumed_empty;

    OP_REQUIRES_OK(ctx, ctx->input("debug_name", &debug_name));
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    OP_REQUIRES_OK(ctx, ctx->input("feature_list_dense_missing_assumed_empty",
                                   &feature_list_dense_missing_assumed_empty));
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("context_dense_keys", &context_dense_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("feature_list_dense_keys",
                                        &feature_list_dense_keys));
    OP_REQUIRES_OK(
        ctx, ctx->input_list("context_sparse_keys", &context_sparse_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("feature_list_sparse_keys",
                                        &feature_list_sparse_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("context_dense_defaults",
                                        &context_dense_defaults));

    std::vector<string> context_dense_keys_t(num_context_dense_);
    std::vector<string> context_sparse_keys_t(num_context_sparse_);
    std::vector<string> feature_list_dense_keys_t(num_feature_list_dense_);
    std::vector<string> feature_list_sparse_keys_t(num_feature_list_sparse_);
    std::unordered_set<string> feature_list_dense_missing_assumed_empty_set;
    CHECK_EQ(context_dense_keys.size(), num_context_dense_);
    CHECK_EQ(context_sparse_keys.size(), num_context_sparse_);
    CHECK_EQ(feature_list_dense_keys.size(), num_feature_list_dense_);
    CHECK_EQ(feature_list_sparse_keys.size(), num_feature_list_sparse_);
    for (int di = 0; di < num_context_dense_; ++di) {
      OP_REQUIRES(ctx,
                  TensorShapeUtils::IsScalar(context_dense_keys[di].shape()),
                  errors::InvalidArgument(
                      "Expected context_dense_keys[", di,
                      "] to be a vector, got shape: ",
                      context_dense_keys[di].shape().DebugString()));
      context_dense_keys_t[di] = context_dense_keys[di].scalar<string>()();
    }
    for (int di = 0; di < num_context_sparse_; ++di) {
      OP_REQUIRES(ctx,
                  TensorShapeUtils::IsScalar(context_sparse_keys[di].shape()),
                  errors::InvalidArgument(
                      "Expected context_sparse_keys[", di,
                      "] to be a vector, got shape: ",
                      context_sparse_keys[di].shape().DebugString()));
      context_sparse_keys_t[di] = context_sparse_keys[di].scalar<string>()();
    }
    for (int di = 0; di < num_feature_list_dense_; ++di) {
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(feature_list_dense_keys[di].shape()),
          errors::InvalidArgument(
              "Expected feature_list_dense_keys[", di,
              "] to be a vector, got shape: ",
              feature_list_dense_keys[di].shape().DebugString()));
      feature_list_dense_keys_t[di] =
          feature_list_dense_keys[di].scalar<string>()();
    }
    for (int di = 0; di < num_feature_list_sparse_; ++di) {
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(feature_list_sparse_keys[di].shape()),
          errors::InvalidArgument(
              "Expected feature_list_sparse_keys[", di,
              "] to be a vector, got shape: ",
              feature_list_sparse_keys[di].shape().DebugString()));
      feature_list_sparse_keys_t[di] =
          feature_list_sparse_keys[di].scalar<string>()();
    }
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(
                 feature_list_dense_missing_assumed_empty->shape()),
        errors::InvalidArgument(
            "Expected feature_list_dense_missing_assumed_empty ",
            "to be a vector, got shape: ",
            feature_list_dense_missing_assumed_empty->shape().DebugString()));
    auto feature_list_dense_missing_assumped_empty_t =
        feature_list_dense_missing_assumed_empty->vec<string>();
    for (int de = 0;
         de < feature_list_dense_missing_assumed_empty->NumElements(); ++de) {
      feature_list_dense_missing_assumed_empty_set.insert(
          feature_list_dense_missing_assumped_empty_t(de));
    }

    bool has_debug_name = (debug_name->NumElements() > 0);
    if (has_debug_name) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(debug_name->shape()),
                  errors::InvalidArgument(
                      "Expected debug_name to be a scalar, got shape: ",
                      debug_name->shape().DebugString()));
    }
    auto debug_name_t = debug_name->scalar<string>();

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(serialized->shape()),
                errors::InvalidArgument(
                    "Expected serialized to be a scalar, got shape: ",
                    serialized->shape().DebugString()));

    OP_REQUIRES(ctx, context_dense_defaults.size() == num_context_dense_,
                errors::InvalidArgument("Expected len(context_dense_defaults) "
                                        "== len(context_dense_keys) but got: ",
                                        context_dense_defaults.size(), " vs. ",
                                        num_context_dense_));

    std::vector<bool> required(num_context_dense_);
    for (int d = 0; d < num_context_dense_; ++d) {
      const Tensor& def_value = context_dense_defaults[d];
      required[d] = (def_value.NumElements() == 0);  // No default provided.

      if (def_value.NumElements() > 0) {
        OP_REQUIRES(
            ctx, def_value.shape() == context_dense_shapes_[d],
            errors::InvalidArgument("def_value[", d, "].shape() == ",
                                    def_value.shape().DebugString(),
                                    " != context_dense_shapes_[", d, "] == ",
                                    context_dense_shapes_[d].DebugString()));
        OP_REQUIRES(
            ctx, def_value.dtype() == context_dense_types_[d],
            errors::InvalidArgument(
                "context_dense_defaults[", d, "].dtype() == ",
                DataTypeString(def_value.dtype()), " != context_dense_types_[",
                d, "] == ", DataTypeString(context_dense_types_[d])));
      }
    }

    auto serialized_t = serialized->scalar<string>();

    OpOutputList context_sparse_indices;
    OpOutputList context_sparse_values;
    OpOutputList context_sparse_shapes;
    OpOutputList context_dense_values;
    OpOutputList feature_list_sparse_indices;
    OpOutputList feature_list_sparse_values;
    OpOutputList feature_list_sparse_shapes;
    OpOutputList feature_list_dense_values;

    OP_REQUIRES_OK(ctx, ctx->output_list("context_sparse_indices",
                                         &context_sparse_indices));
    OP_REQUIRES_OK(
        ctx, ctx->output_list("context_sparse_values", &context_sparse_values));
    OP_REQUIRES_OK(
        ctx, ctx->output_list("context_sparse_shapes", &context_sparse_shapes));
    OP_REQUIRES_OK(
        ctx, ctx->output_list("context_dense_values", &context_dense_values));
    OP_REQUIRES_OK(ctx, ctx->output_list("context_sparse_indices",
                                         &context_sparse_indices));
    OP_REQUIRES_OK(ctx, ctx->output_list("feature_list_sparse_indices",
                                         &feature_list_sparse_indices));
    OP_REQUIRES_OK(ctx, ctx->output_list("feature_list_sparse_values",
                                         &feature_list_sparse_values));
    OP_REQUIRES_OK(ctx, ctx->output_list("feature_list_sparse_shapes",
                                         &feature_list_sparse_shapes));
    OP_REQUIRES_OK(ctx, ctx->output_list("feature_list_dense_values",
                                         &feature_list_dense_values));

    SequenceExample ex;
    OP_REQUIRES(
        ctx, ParseProtoUnlimited(&ex, serialized_t()),
        errors::InvalidArgument("Could not parse example input, value: '",
                                serialized_t(), "'"));

    const string& name = (has_debug_name) ? debug_name_t() : "<unknown>";
    const Features& context = ex.context();
    const auto& context_dict = context.feature();

    // Context Dense -----------------------------------------------------------

    // Preallocate context_dense_values, since we know their sizes
    for (int d = 0; d < num_context_dense_; ++d) {
      TensorShape out_shape;
      for (const int dim : context_dense_shapes_[d].dim_sizes())
        out_shape.AddDim(dim);
      Tensor* out = nullptr;
      context_dense_values.allocate(d, out_shape, &out);
    }

    for (int d = 0; d < num_context_dense_; ++d) {
      const string& key = context_dense_keys_t[d];
      const DataType& dtype = context_dense_types_[d];
      const TensorShape& shape = context_dense_shapes_[d];

      const auto& feature_found = context_dict.find(key);
      OP_REQUIRES(
          ctx, (feature_found != context_dict.end()) || !required[d],
          errors::InvalidArgument("Name: ", name, ", Context feature '", key,
                                  "' is required but could not be found."));
      if (feature_found != context_dict.end()) {
        const Feature& f = feature_found->second;
        bool types_match;
        OP_REQUIRES_OK(ctx, CheckTypesMatch(f, dtype, &types_match));
        OP_REQUIRES(
            ctx, types_match,
            errors::InvalidArgument("Name: ", name, ", Context feature: ", key,
                                    ".  Data types don't match. ",
                                    "Expected type: ", DataTypeString(dtype),
                                    "  Feature is: ", f.DebugString()));

        OP_REQUIRES_OK(ctx, FeatureDenseCopy(0, name, key, dtype, shape, f,
                                             context_dense_values[d]));
      } else {
        RowDenseCopy(0, dtype, context_dense_defaults[d],
                     context_dense_values[d]);
      }
    }

    // Context Sparse ----------------------------------------------------------
    for (int d = 0; d < num_context_sparse_; ++d) {
      const string& key = context_sparse_keys_t[d];
      const DataType& dtype = context_sparse_types_[d];

      const auto& feature_found = context_dict.find(key);
      bool feature_has_data =  // Found key & data type is set
          (feature_found != context_dict.end() &&
           (feature_found->second.kind_case() != Feature::KIND_NOT_SET));

      if (feature_has_data) {
        const Feature& f = feature_found->second;
        bool types_match;
        OP_REQUIRES_OK(ctx, CheckTypesMatch(f, dtype, &types_match));
        OP_REQUIRES(
            ctx, types_match,
            errors::InvalidArgument("Name: ", name, ", Context feature: ", key,
                                    ".  Data types don't match. ",
                                    "Expected type: ", DataTypeString(dtype),
                                    "  Feature is: ", f.DebugString()));

        Tensor feature_values = FeatureSparseCopy(0, key, dtype, f);
        const int64 num_elements = feature_values.NumElements();
        TensorShape indices_shape({num_elements, 1});
        Tensor* sp_indices_d = nullptr;
        Tensor* sp_shape_d = nullptr;
        context_sparse_indices.allocate(d, indices_shape, &sp_indices_d);
        context_sparse_values.set(d, feature_values);
        context_sparse_shapes.allocate(d, TensorShape({1}), &sp_shape_d);
        auto shape_t = sp_shape_d->vec<int64>();
        shape_t(0) = num_elements;
        auto indices_t = sp_indices_d->matrix<int64>();
        std::iota(indices_t.data(), indices_t.data() + num_elements, 0);
      } else {
        TensorShape indices_shape({0, 1});
        TensorShape values_shape({0});
        Tensor* sp_indices_d = nullptr;
        Tensor* sp_values_d = nullptr;
        Tensor* sp_shape_d = nullptr;
        context_sparse_indices.allocate(d, indices_shape, &sp_indices_d);
        context_sparse_values.allocate(d, values_shape, &sp_values_d);
        context_sparse_shapes.allocate(d, TensorShape({1}), &sp_shape_d);
        auto shape_t = sp_shape_d->vec<int64>();
        shape_t(0) = 0;
      }
    }

    // Feature List Dense ------------------------------------------------------

    // Preallocate context_dense_values, since we can infer their
    // sizes
    const FeatureLists& feature_lists = ex.feature_lists();
    const auto& feature_list_dict = feature_lists.feature_list();
    FeatureList empty_feature_list;  // Placeholder for missing FLs

    for (int d = 0; d < num_feature_list_dense_; ++d) {
      const string& key = feature_list_dense_keys_t[d];
      const DataType& dtype = feature_list_dense_types_[d];
      const TensorShape& shape = feature_list_dense_shapes_[d];

      const auto& feature_list_found = feature_list_dict.find(key);
      bool feature_list_missing =
          (feature_list_found == feature_list_dict.end());
      bool feature_list_allowed_missing =
          (feature_list_dense_missing_assumed_empty_set.count(key) > 0);

      OP_REQUIRES(
          ctx, !feature_list_missing || feature_list_allowed_missing,
          errors::InvalidArgument("Name: ", name, ", Feature list '", key,
                                  "' is required but could not be found.  "
                                  "Did you mean to include it in "
                                  "feature_list_dense_missing_assumed_empty or "
                                  "feature_list_dense_defaults?"));

      TensorShape out_shape;
      const FeatureList& fl = (feature_list_missing)
                                  ? empty_feature_list
                                  : feature_list_found->second;
      out_shape.AddDim(fl.feature_size());
      for (const int dim : feature_list_dense_shapes_[d].dim_sizes()) {
        out_shape.AddDim(dim);
      }
      Tensor* out = nullptr;
      feature_list_dense_values.allocate(d, out_shape, &out);

      for (int64 t = 0; t < fl.feature_size(); ++t) {
        const Feature& f = fl.feature(t);
        bool types_match;
        OP_REQUIRES_OK(ctx, CheckTypesMatch(f, dtype, &types_match));
        OP_REQUIRES(
            ctx, types_match,
            errors::InvalidArgument(
                "Name: ", name, ", Feature list: ", key, ", Index: ", t,
                ".  Data types don't match. ", "Expected type: ",
                DataTypeString(dtype), "  Feature is: ", f.DebugString()));
        OP_REQUIRES_OK(ctx, FeatureDenseCopy(t, name, key, dtype, shape, f,
                                             feature_list_dense_values[d]));
      }
    }

    // Feature List Sparse -----------------------------------------------------
    for (int d = 0; d < num_feature_list_sparse_; ++d) {
      const string& key = feature_list_sparse_keys_t[d];
      const DataType& dtype = feature_list_sparse_types_[d];

      const auto& feature_list_found = feature_list_dict.find(key);
      bool feature_list_has_data =  // Found key
          (feature_list_found != feature_list_dict.end());

      std::vector<Tensor> sparse_values_tmp;
      int64 feature_list_size = 0;
      if (feature_list_has_data) {
        const FeatureList& fl = feature_list_found->second;
        feature_list_size = fl.feature_size();
        for (int64 t = 0; t < feature_list_size; ++t) {
          const Feature& f = fl.feature(t);
          bool types_match;
          OP_REQUIRES_OK(ctx, CheckTypesMatch(f, dtype, &types_match));
          OP_REQUIRES(
              ctx, types_match,
              errors::InvalidArgument(
                  "Name: ", name, ", Feature List: ", key, ", Index: ", t,
                  ".  Data types don't match. ", "Expected type: ",
                  DataTypeString(dtype), "  Feature is: ", f.DebugString()));
          sparse_values_tmp.push_back(FeatureSparseCopy(t, key, dtype, f));
        }
      } else {
        sparse_values_tmp.push_back(Tensor(dtype, TensorShape({0})));
      }

      int64 total_num_features = 0;
      int64 max_num_features = 0;
      for (int t = 0; t < feature_list_size; ++t) {
        const Tensor& v = sparse_values_tmp[t];
        const int64 num_elements = v.shape().num_elements();
        total_num_features += num_elements;
        max_num_features = std::max(max_num_features, num_elements);
      }

      TensorShape indices_shape({total_num_features, 2});
      TensorShape values_shape({total_num_features});
      Tensor* sp_indices_d = nullptr;
      Tensor* sp_values_d = nullptr;
      Tensor* sp_shape_d = nullptr;
      feature_list_sparse_indices.allocate(d, indices_shape, &sp_indices_d);
      feature_list_sparse_values.allocate(d, values_shape, &sp_values_d);
      feature_list_sparse_shapes.allocate(d, TensorShape({2}), &sp_shape_d);
      auto shape_t = sp_shape_d->vec<int64>();
      shape_t(0) = feature_list_size;
      shape_t(1) = max_num_features;

      int64 offset = 0;

      for (int t = 0; t < feature_list_size; ++t) {
        const int64 num_elements = CopyIntoSparseTensor(
            sparse_values_tmp[t], t, offset, sp_indices_d, sp_values_d);
        offset += num_elements;
      }
    }
  }

 protected:
  int64 num_context_sparse_;
  int64 num_context_dense_;
  int64 num_feature_list_sparse_;
  int64 num_feature_list_dense_;
  std::vector<DataType> context_sparse_types_;
  std::vector<DataType> context_dense_types_;
  std::vector<TensorShape> context_dense_shapes_;
  std::vector<DataType> feature_list_sparse_types_;
  std::vector<DataType> feature_list_dense_types_;
  std::vector<TensorShape> feature_list_dense_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("ParseSingleSequenceExample").Device(DEVICE_CPU),
                        SingleSequenceExampleParserOp);

class DecodeJSONExampleOp : public OpKernel {
 public:
  explicit DecodeJSONExampleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    resolver_.reset(protobuf::util::NewTypeResolverForDescriptorPool(
        "type.googleapis.com", protobuf::DescriptorPool::generated_pool()));
  }

  void Compute(OpKernelContext* ctx) {
    const Tensor* json_examples;
    OP_REQUIRES_OK(ctx, ctx->input("json_examples", &json_examples));
    Tensor* binary_examples;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("binary_examples", json_examples->shape(),
                                  &binary_examples));

    for (int i = 0; i < json_examples->NumElements(); ++i) {
      const string& json_example = json_examples->flat<string>()(i);
      auto status = protobuf::util::JsonToBinaryString(
          resolver_.get(), "type.googleapis.com/tensorflow.Example",
          json_example, &binary_examples->flat<string>()(i));
      OP_REQUIRES(ctx, status.ok(),
                  errors::InvalidArgument("Error while parsing JSON: ",
                                          string(status.error_message())));
    }
  }

 private:
  std::unique_ptr<protobuf::util::TypeResolver> resolver_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeJSONExample").Device(DEVICE_CPU),
                        DecodeJSONExampleOp);

}  // namespace tensorflow
