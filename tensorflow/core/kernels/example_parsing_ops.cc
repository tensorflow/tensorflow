/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <numeric>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb_text.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/example_proto_fast_parsing.h"
#include "tensorflow/core/util/example_proto_helper.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

class ExampleParserOp : public OpKernel {
 public:
  explicit ExampleParserOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, attrs_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* names;
    const Tensor* serialized;
    OpInputList dense_keys;
    OpInputList sparse_keys;
    OpInputList dense_defaults;

    // Grab the input list arguments.
    OP_REQUIRES_OK(ctx, ctx->input("names", &names));
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_keys", &dense_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("sparse_keys", &sparse_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_defaults", &dense_defaults));

    std::vector<string> dense_keys_t(attrs_.num_dense);
    std::vector<string> sparse_keys_t(attrs_.num_sparse);

    // Check that the input list sizes match the attribute declared sizes.
    CHECK_EQ(dense_keys.size(), attrs_.num_dense);
    CHECK_EQ(sparse_keys.size(), attrs_.num_sparse);

    // Copy from OpInputList to std::vector<string>.
    for (int di = 0; di < attrs_.num_dense; ++di) {
      dense_keys_t[di] = dense_keys[di].scalar<string>()();
    }
    for (int di = 0; di < attrs_.num_sparse; ++di) {
      sparse_keys_t[di] = sparse_keys[di].scalar<string>()();
    }

    if (names->NumElements() > 0) {
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

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(serialized->shape()),
                errors::InvalidArgument(
                    "Expected serialized to be a vector, got shape: ",
                    serialized->shape().DebugString()));
    OP_REQUIRES(ctx, dense_defaults.size() == attrs_.num_dense,
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_defaults.size(), " vs. ", attrs_.num_dense));

    for (int d = 0; d < static_cast<int>(attrs_.num_dense); ++d) {
      const Tensor& def_value = dense_defaults[d];
      if (def_value.NumElements() > 0) {
        OP_REQUIRES(ctx,
                    attrs_.dense_shapes[d].IsCompatibleWith(def_value.shape()),
                    errors::InvalidArgument(
                        "def_value[", d,
                        "].shape() == ", def_value.shape().DebugString(),
                        " is not compatible with dense_shapes_[", d,
                        "] == ", attrs_.dense_shapes[d].DebugString()));
        OP_REQUIRES(ctx, def_value.dtype() == attrs_.dense_types[d],
                    errors::InvalidArgument(
                        "dense_defaults[", d, "].dtype() == ",
                        DataTypeString(def_value.dtype()), " != dense_types_[",
                        d, "] == ", DataTypeString(attrs_.dense_types[d])));
      }
    }

    example::Result result;

    example::FastParseExampleConfig config;
    for (int d = 0; d < attrs_.num_dense; ++d) {
      config.dense.push_back({dense_keys_t[d], attrs_.dense_types[d],
                              attrs_.dense_shapes[d], dense_defaults[d],
                              attrs_.elements_per_stride[d]});
    }
    for (int d = 0; d < attrs_.num_sparse; ++d) {
      config.sparse.push_back({sparse_keys_t[d], attrs_.sparse_types[d]});
    }

    auto serialized_t = serialized->flat<string>();
    auto names_t = names->flat<string>();
    gtl::ArraySlice<string> slice(serialized_t.data(), serialized_t.size());
    gtl::ArraySlice<string> names_slice(names_t.data(), names_t.size());

    OP_REQUIRES_OK(
        ctx,
        FastParseExample(
            config, slice, names_slice,
            ctx->device()->tensorflow_cpu_worker_threads()->workers, &result));

    OpOutputList dense_values;
    OpOutputList sparse_indices;
    OpOutputList sparse_values;
    OpOutputList sparse_shapes;
    OP_REQUIRES_OK(ctx, ctx->output_list("dense_values", &dense_values));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_indices", &sparse_indices));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_values", &sparse_values));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_shapes", &sparse_shapes));
    for (int d = 0; d < attrs_.num_dense; ++d) {
      dense_values.set(d, result.dense_values[d]);
    }
    for (int d = 0; d < attrs_.num_sparse; ++d) {
      sparse_indices.set(d, result.sparse_indices[d]);
      sparse_values.set(d, result.sparse_values[d]);
      sparse_shapes.set(d, result.sparse_shapes[d]);
    }
  }

 protected:
  ParseSingleExampleAttrs attrs_;
};

REGISTER_KERNEL_BUILDER(Name("ParseExample").Device(DEVICE_CPU),
                        ExampleParserOp);

class SingleSequenceExampleParserOp : public OpKernel {
 public:
  explicit SingleSequenceExampleParserOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, attrs_.Init(ctx));
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

    std::vector<string> context_dense_keys_t(attrs_.num_context_dense);
    std::vector<string> context_sparse_keys_t(attrs_.num_context_sparse);
    std::vector<string> feature_list_dense_keys_t(
        attrs_.num_feature_list_dense);
    std::vector<string> feature_list_sparse_keys_t(
        attrs_.num_feature_list_sparse);
    std::unordered_set<string> feature_list_dense_missing_assumed_empty_set;
    CHECK_EQ(context_dense_keys.size(), attrs_.num_context_dense);
    CHECK_EQ(context_sparse_keys.size(), attrs_.num_context_sparse);
    CHECK_EQ(feature_list_dense_keys.size(), attrs_.num_feature_list_dense);
    CHECK_EQ(feature_list_sparse_keys.size(), attrs_.num_feature_list_sparse);
    for (int di = 0; di < attrs_.num_context_dense; ++di) {
      OP_REQUIRES(ctx,
                  TensorShapeUtils::IsScalar(context_dense_keys[di].shape()),
                  errors::InvalidArgument(
                      "Expected context_dense_keys[", di,
                      "] to be a vector, got shape: ",
                      context_dense_keys[di].shape().DebugString()));
      context_dense_keys_t[di] = context_dense_keys[di].scalar<string>()();
    }
    for (int di = 0; di < attrs_.num_context_sparse; ++di) {
      OP_REQUIRES(ctx,
                  TensorShapeUtils::IsScalar(context_sparse_keys[di].shape()),
                  errors::InvalidArgument(
                      "Expected context_sparse_keys[", di,
                      "] to be a vector, got shape: ",
                      context_sparse_keys[di].shape().DebugString()));
      context_sparse_keys_t[di] = context_sparse_keys[di].scalar<string>()();
    }
    for (int di = 0; di < attrs_.num_feature_list_dense; ++di) {
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(feature_list_dense_keys[di].shape()),
          errors::InvalidArgument(
              "Expected feature_list_dense_keys[", di,
              "] to be a vector, got shape: ",
              feature_list_dense_keys[di].shape().DebugString()));
      feature_list_dense_keys_t[di] =
          feature_list_dense_keys[di].scalar<string>()();
    }
    for (int di = 0; di < attrs_.num_feature_list_sparse; ++di) {
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

    OP_REQUIRES(ctx, context_dense_defaults.size() == attrs_.num_context_dense,
                errors::InvalidArgument("Expected len(context_dense_defaults) "
                                        "== len(context_dense_keys) but got: ",
                                        context_dense_defaults.size(), " vs. ",
                                        attrs_.num_context_dense));

    std::vector<bool> required(attrs_.num_context_dense);
    for (int d = 0; d < attrs_.num_context_dense; ++d) {
      const Tensor& def_value = context_dense_defaults[d];
      required[d] = (def_value.NumElements() == 0);  // No default provided.

      if (def_value.NumElements() > 0) {
        OP_REQUIRES(
            ctx, def_value.shape() == attrs_.context_dense_shapes[d],
            errors::InvalidArgument(
                "def_value[", d, "].shape() == ",
                def_value.shape().DebugString(), " != context_dense_shapes_[",
                d, "] == ", attrs_.context_dense_shapes[d].DebugString()));
        OP_REQUIRES(
            ctx, def_value.dtype() == attrs_.context_dense_types[d],
            errors::InvalidArgument(
                "context_dense_defaults[", d, "].dtype() == ",
                DataTypeString(def_value.dtype()), " != context_dense_types_[",
                d, "] == ", DataTypeString(attrs_.context_dense_types[d])));
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
    for (int d = 0; d < attrs_.num_context_dense; ++d) {
      TensorShape out_shape;
      for (const int dim : attrs_.context_dense_shapes[d].dim_sizes())
        out_shape.AddDim(dim);
      Tensor* out = nullptr;
      context_dense_values.allocate(d, out_shape, &out);
    }

    for (int d = 0; d < attrs_.num_context_dense; ++d) {
      const string& key = context_dense_keys_t[d];
      const DataType& dtype = attrs_.context_dense_types[d];
      const TensorShape& shape = attrs_.context_dense_shapes[d];

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
                                    "  Feature is: ", ProtoDebugString(f)));

        OP_REQUIRES_OK(ctx, FeatureDenseCopy(0, name, key, dtype, shape, f,
                                             context_dense_values[d]));
      } else {
        RowDenseCopy(0, dtype, context_dense_defaults[d],
                     context_dense_values[d]);
      }
    }

    // Context Sparse ----------------------------------------------------------
    for (int d = 0; d < attrs_.num_context_sparse; ++d) {
      const string& key = context_sparse_keys_t[d];
      const DataType& dtype = attrs_.context_sparse_types[d];

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
                                    "  Feature is: ", ProtoDebugString(f)));

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

    for (int d = 0; d < attrs_.num_feature_list_dense; ++d) {
      const string& key = feature_list_dense_keys_t[d];
      const DataType& dtype = attrs_.feature_list_dense_types[d];
      const TensorShape& shape = attrs_.feature_list_dense_shapes[d];

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
      for (const int dim : attrs_.feature_list_dense_shapes[d].dim_sizes()) {
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
                DataTypeString(dtype), "  Feature is: ", ProtoDebugString(f)));
        OP_REQUIRES_OK(ctx, FeatureDenseCopy(t, name, key, dtype, shape, f,
                                             feature_list_dense_values[d]));
      }
    }

    // Feature List Sparse -----------------------------------------------------
    for (int d = 0; d < attrs_.num_feature_list_sparse; ++d) {
      const string& key = feature_list_sparse_keys_t[d];
      const DataType& dtype = attrs_.feature_list_sparse_types[d];

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
              ctx, f.kind_case() == Feature::KIND_NOT_SET || types_match,
              errors::InvalidArgument("Name: ", name, ", Feature List: ", key,
                                      ", Index: ", t,
                                      ".  Data types don't match. ",
                                      "Expected type: ", DataTypeString(dtype),
                                      "  Feature is: ", ProtoDebugString(f)));
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
  ParseSingleSequenceExampleAttrs attrs_;
};

REGISTER_KERNEL_BUILDER(Name("ParseSingleSequenceExample").Device(DEVICE_CPU),
                        SingleSequenceExampleParserOp);

#ifndef IS_MOBILE_PLATFORM
// when using lite protos on mobile, decoding JSON is not available.

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
#endif

}  // namespace tensorflow
