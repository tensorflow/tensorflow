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

#include "absl/base/call_once.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/example_proto_fast_parsing.h"
#include "tensorflow/core/util/example_proto_helper.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {
constexpr char kParseExampleV2[] = "ParseExampleV2";
constexpr char kParseSequenceExampleV2[] = "ParseSequenceExampleV2";
}  // namespace

// Note: this kernel is used by both the ParseExample op and the ParseExampleV2
// op.  It automatically determines which op was used by checking if the
// "ragged_value_types" attribute exists.
class ParseExampleOp : public OpKernel {
 public:
  explicit ParseExampleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), op_version_(ctx->def().op() == kParseExampleV2 ? 2 : 1) {
    OP_REQUIRES_OK(ctx, attrs_.Init(ctx, op_version_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* names;
    const Tensor* serialized;
    std::vector<StringPiece> dense_keys_t;
    std::vector<StringPiece> sparse_keys_t;
    std::vector<StringPiece> ragged_keys_t;
    OpInputList dense_defaults;

    // Grab the inputs.
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    OP_REQUIRES_OK(ctx, ctx->input("names", &names));
    if (op_version_ == 2) {
      OP_REQUIRES_OK(ctx, GetTensorKeys(ctx, "dense_keys", &dense_keys_t));
      OP_REQUIRES_OK(ctx, GetTensorKeys(ctx, "sparse_keys", &sparse_keys_t));
      OP_REQUIRES_OK(ctx, GetTensorKeys(ctx, "ragged_keys", &ragged_keys_t));
    } else {
      OP_REQUIRES_OK(ctx, GetInputListKeys(ctx, "dense_keys", &dense_keys_t));
      OP_REQUIRES_OK(ctx, GetInputListKeys(ctx, "sparse_keys", &sparse_keys_t));
    }
    absl::call_once(flag_, [&dense_keys_t, &sparse_keys_t, &ragged_keys_t]() {
      metrics::RecordParseDenseFeature(dense_keys_t.size());
      metrics::RecordParseSparseFeature(sparse_keys_t.size());
      metrics::RecordParseRaggedFeature(ragged_keys_t.size());
    });
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_defaults", &dense_defaults));

    // Validate input tensor shapes.
    OP_REQUIRES_OK(
        ctx, CheckInputShapes(serialized, names, dense_defaults, dense_keys_t,
                              sparse_keys_t, ragged_keys_t));

    example::FastParseExampleConfig config =
        MakeConfig(dense_keys_t, sparse_keys_t, ragged_keys_t, dense_defaults);

    example::Result result;
    if (TensorShapeUtils::IsVector(serialized->shape())) {
      OP_REQUIRES_OK(
          ctx, ParseExampleVector(config, serialized, names, ctx, &result));
    } else {
      OP_REQUIRES_OK(ctx, ParseExampleScalar(config, serialized, ctx, &result));
    }
    OP_REQUIRES_OK(ctx, WriteOutput(result, ctx));
  }

 protected:
  // Copies keys from tensor to std::vector<string>.
  Status GetTensorKeys(OpKernelContext* ctx, StringPiece input_name,
                       std::vector<StringPiece>* keys) const {
    const Tensor* key_t;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &key_t));
    keys->reserve(key_t->NumElements());
    auto keys_flat = key_t->flat<tstring>();
    for (int i = 0; i < keys_flat.size(); ++i) {
      keys->push_back(keys_flat(i));
    }
    return Status::OK();
  }

  // Copies keys from OpInputList of scalar to std::vector<string>.
  Status GetInputListKeys(OpKernelContext* ctx, StringPiece input_name,
                          std::vector<StringPiece>* keys) const {
    OpInputList key_list;
    TF_RETURN_IF_ERROR(ctx->input_list(input_name, &key_list));
    keys->reserve(key_list.size());
    for (const auto& key : key_list) {
      keys->push_back(key.scalar<tstring>()());
    }
    return Status::OK();
  }

  // Validates the shapes of input tensors.
  Status CheckInputShapes(const Tensor* serialized, const Tensor* names,
                          const OpInputList& dense_defaults,
                          const std::vector<StringPiece>& dense_keys_t,
                          const std::vector<StringPiece>& sparse_keys_t,
                          const std::vector<StringPiece>& ragged_keys_t) const {
    if (op_version_ == 2) {
      if (TensorShapeUtils::IsMatrixOrHigher(serialized->shape())) {
        return errors::InvalidArgument(
            "Expected serialized to be a scalar or vector, got shape: ",
            serialized->shape().DebugString());
      }
    } else {
      if (!TensorShapeUtils::IsVector(serialized->shape())) {
        return errors::InvalidArgument(
            "Expected serialized to be a vector, got shape: ",
            serialized->shape().DebugString());
      }
    }
    if (names->NumElements() > 0 && names->shape() != serialized->shape()) {
      return errors::InvalidArgument(
          "Expected names have the same shape as serialized: name.shape=",
          names->shape().DebugString(),
          ", serialized.shape=", serialized->shape().DebugString());
    }
    if (op_version_ == 2) {
      if (dense_keys_t.size() != attrs_.num_dense) {
        return errors::InvalidArgument(
            "Expected len(dense_keys) == len(dense_types) but got: ",
            dense_keys_t.size(), " vs. ", attrs_.num_dense);
      }
      if (sparse_keys_t.size() != attrs_.num_sparse) {
        return errors::InvalidArgument(
            "Expected len(sparse_keys) == num_sparse but got: ",
            sparse_keys_t.size(), " vs. ", attrs_.num_sparse);
      }
      if (ragged_keys_t.size() != attrs_.num_ragged) {
        return errors::InvalidArgument(
            "Expected len(ragged_keys) == len(ragged_value_types) but got: ",
            ragged_keys_t.size(), " vs. ", attrs_.num_ragged);
      }
    }

    if (dense_defaults.size() != attrs_.num_dense) {
      return errors::InvalidArgument(
          "Expected len(dense_defaults) == len(dense_keys) but got: ",
          dense_defaults.size(), " vs. ", attrs_.num_dense);
    }

    for (int d = 0; d < static_cast<int>(attrs_.num_dense); ++d) {
      const Tensor& def_value = dense_defaults[d];
      if (attrs_.variable_length[d]) {
        if (def_value.NumElements() != 1) {
          return errors::InvalidArgument(
              "dense_shape[", d, "] is a variable length shape: ",
              attrs_.dense_shapes[d].DebugString(),
              ", therefore "
              "def_value[",
              d,
              "] must contain a single element ("
              "the padding element).  But its shape is: ",
              def_value.shape().DebugString());
        }
      } else if (def_value.NumElements() > 0) {
        if (!attrs_.dense_shapes[d].IsCompatibleWith(def_value.shape())) {
          return errors::InvalidArgument(
              "def_value[", d, "].shape() == ", def_value.shape().DebugString(),
              " is not compatible with dense_shapes_[", d,
              "] == ", attrs_.dense_shapes[d].DebugString());
        }
      }
      if (def_value.dtype() != attrs_.dense_types[d]) {
        return errors::InvalidArgument(
            "dense_defaults[", d,
            "].dtype() == ", DataTypeString(def_value.dtype()),
            " != dense_types_[", d,
            "] == ", DataTypeString(attrs_.dense_types[d]));
      }
    }
    return Status::OK();
  }

  // Populates the FastParseExampleConfig from keys & defaults.
  example::FastParseExampleConfig MakeConfig(
      const std::vector<StringPiece>& dense_keys_t,
      const std::vector<StringPiece>& sparse_keys_t,
      const std::vector<StringPiece>& ragged_keys_t,
      const OpInputList& dense_defaults) const {
    example::FastParseExampleConfig config;
    config.dense.reserve(attrs_.num_dense);
    for (int d = 0; d < attrs_.num_dense; ++d) {
      config.dense.emplace_back(dense_keys_t[d], attrs_.dense_types[d],
                                attrs_.dense_shapes[d], dense_defaults[d],
                                attrs_.variable_length[d],
                                attrs_.elements_per_stride[d]);
    }
    config.sparse.reserve(attrs_.num_sparse);
    for (int d = 0; d < attrs_.num_sparse; ++d) {
      config.sparse.emplace_back(sparse_keys_t[d], attrs_.sparse_types[d]);
    }
    config.sparse.reserve(attrs_.num_ragged);
    for (int d = 0; d < attrs_.num_ragged; ++d) {
      config.ragged.emplace_back(ragged_keys_t[d], attrs_.ragged_value_types[d],
                                 attrs_.ragged_split_types[d]);
    }
    return config;
  }

  // Parses a single example.
  Status ParseExampleScalar(const example::FastParseExampleConfig& config,
                            const Tensor* serialized, OpKernelContext* ctx,
                            example::Result* result) const {
    const tstring& serialized_proto = serialized->scalar<tstring>()();
    return FastParseSingleExample(config, serialized_proto, result);
  }

  // Parses a vector of examples.
  Status ParseExampleVector(const example::FastParseExampleConfig& config,
                            const Tensor* serialized, const Tensor* names,
                            OpKernelContext* ctx,
                            example::Result* result) const {
    auto serialized_t = serialized->flat<tstring>();
    auto names_t = names->flat<tstring>();
    gtl::ArraySlice<tstring> slice(serialized_t.data(), serialized_t.size());
    gtl::ArraySlice<tstring> names_slice(names_t.data(), names_t.size());
    return FastParseExample(
        config, slice, names_slice,
        ctx->device()->tensorflow_cpu_worker_threads()->workers, result);
  }

  Status WriteOutput(const example::Result& result,
                     OpKernelContext* ctx) const {
    OpOutputList dense_values;
    OpOutputList sparse_indices;
    OpOutputList sparse_values;
    OpOutputList sparse_shapes;
    TF_RETURN_IF_ERROR(ctx->output_list("dense_values", &dense_values));
    TF_RETURN_IF_ERROR(ctx->output_list("sparse_indices", &sparse_indices));
    TF_RETURN_IF_ERROR(ctx->output_list("sparse_values", &sparse_values));
    TF_RETURN_IF_ERROR(ctx->output_list("sparse_shapes", &sparse_shapes));
    for (int d = 0; d < attrs_.num_dense; ++d) {
      dense_values.set(d, result.dense_values[d]);
    }
    for (int d = 0; d < attrs_.num_sparse; ++d) {
      sparse_indices.set(d, result.sparse_indices[d]);
      sparse_values.set(d, result.sparse_values[d]);
      sparse_shapes.set(d, result.sparse_shapes[d]);
    }
    if (op_version_ == 2) {
      OpOutputList ragged_values;
      OpOutputList ragged_splits;
      TF_RETURN_IF_ERROR(ctx->output_list("ragged_values", &ragged_values));
      TF_RETURN_IF_ERROR(ctx->output_list("ragged_row_splits", &ragged_splits));
      for (int d = 0; d < attrs_.num_ragged; ++d) {
        ragged_values.set(d, result.ragged_values[d]);
        ragged_splits.set(d, result.ragged_splits[d]);
      }
    }
    return Status::OK();
  }

  ParseExampleAttrs attrs_;
  int op_version_;
  absl::once_flag flag_;
};

REGISTER_KERNEL_BUILDER(Name("ParseExample").Device(DEVICE_CPU),
                        ParseExampleOp);
REGISTER_KERNEL_BUILDER(Name("ParseExampleV2").Device(DEVICE_CPU),
                        ParseExampleOp);

class ParseSingleExampleOp : public OpKernel {
 public:
  explicit ParseSingleExampleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, attrs_.Init(ctx));
    metrics::RecordParseDenseFeature(attrs_.dense_keys.size());
    metrics::RecordParseSparseFeature(attrs_.sparse_keys.size());
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* serialized;
    OpInputList dense_defaults;

    // Grab the input list arguments.
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_defaults", &dense_defaults));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(serialized->shape()),
                errors::InvalidArgument(
                    "Expected serialized to be a scalar, got shape: ",
                    serialized->shape().DebugString()));
    OP_REQUIRES(ctx, dense_defaults.size() == attrs_.dense_keys.size(),
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_defaults.size(), " vs. ", attrs_.dense_keys.size()));

    for (size_t d = 0; d < attrs_.dense_keys.size(); ++d) {
      const Tensor& def_value = dense_defaults[d];
      if (attrs_.variable_length[d]) {
        OP_REQUIRES(ctx, def_value.NumElements() == 1,
                    errors::InvalidArgument(
                        "dense_shape[", d, "] is a variable length shape: ",
                        attrs_.dense_shapes[d].DebugString(),
                        ", therefore "
                        "def_value[",
                        d,
                        "] must contain a single element ("
                        "the padding element).  But its shape is: ",
                        def_value.shape().DebugString()));
      } else if (def_value.NumElements() > 0) {
        OP_REQUIRES(ctx,
                    attrs_.dense_shapes[d].IsCompatibleWith(def_value.shape()),
                    errors::InvalidArgument(
                        "def_value[", d,
                        "].shape() == ", def_value.shape().DebugString(),
                        " is not compatible with dense_shapes_[", d,
                        "] == ", attrs_.dense_shapes[d].DebugString()));
      }
      OP_REQUIRES(ctx, def_value.dtype() == attrs_.dense_types[d],
                  errors::InvalidArgument(
                      "dense_defaults[", d, "].dtype() == ",
                      DataTypeString(def_value.dtype()), " != dense_types_[", d,
                      "] == ", DataTypeString(attrs_.dense_types[d])));
    }

    example::Result result;

    // TODO(mrry): Build the configuration once and cache it.
    example::FastParseExampleConfig config;
    for (int d = 0; d < attrs_.dense_keys.size(); ++d) {
      config.dense.push_back({attrs_.dense_keys[d], attrs_.dense_types[d],
                              attrs_.dense_shapes[d], dense_defaults[d],
                              attrs_.variable_length[d],
                              attrs_.elements_per_stride[d]});
    }
    for (int d = 0; d < attrs_.sparse_keys.size(); ++d) {
      config.sparse.push_back({attrs_.sparse_keys[d], attrs_.sparse_types[d]});
    }

    const tstring& serialized_proto = serialized->scalar<tstring>()();

    OP_REQUIRES_OK(ctx,
                   FastParseSingleExample(config, serialized_proto, &result));

    OpOutputList dense_values;
    OpOutputList sparse_indices;
    OpOutputList sparse_values;
    OpOutputList sparse_shapes;
    OP_REQUIRES_OK(ctx, ctx->output_list("dense_values", &dense_values));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_indices", &sparse_indices));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_values", &sparse_values));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_shapes", &sparse_shapes));
    for (int d = 0; d < attrs_.dense_keys.size(); ++d) {
      dense_values.set(d, result.dense_values[d]);
    }
    for (int d = 0; d < attrs_.sparse_keys.size(); ++d) {
      sparse_indices.set(d, result.sparse_indices[d]);
      sparse_values.set(d, result.sparse_values[d]);
      sparse_shapes.set(d, result.sparse_shapes[d]);
    }
  }

 protected:
  ParseSingleExampleAttrs attrs_;
};

REGISTER_KERNEL_BUILDER(Name("ParseSingleExample").Device(DEVICE_CPU),
                        ParseSingleExampleOp);

class ParseSequenceExampleOp : public OpKernel {
 public:
  explicit ParseSequenceExampleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        op_version_(ctx->def().op() == kParseSequenceExampleV2 ? 2 : 1) {
    OP_REQUIRES_OK(ctx, attrs_.Init(ctx, op_version_));
    metrics::RecordParseDenseFeature(attrs_.context_dense_keys.size() +
                                     attrs_.feature_list_dense_keys.size());
    metrics::RecordParseSparseFeature(attrs_.context_sparse_keys.size() +
                                      attrs_.feature_list_sparse_keys.size());
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* debug_name;
    const Tensor* serialized;
    OpInputList context_dense_defaults;

    OP_REQUIRES_OK(ctx, ctx->input("debug_name", &debug_name));
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    OP_REQUIRES_OK(ctx, ctx->input_list("context_dense_defaults",
                                        &context_dense_defaults));
    const Tensor* context_dense_keys = nullptr;
    const Tensor* context_sparse_keys = nullptr;
    const Tensor* context_ragged_keys = nullptr;
    const Tensor* feature_list_dense_keys = nullptr;
    const Tensor* feature_list_sparse_keys = nullptr;
    const Tensor* feature_list_ragged_keys = nullptr;
    const Tensor* feature_list_dense_missing_assumed_empty = nullptr;
    if (op_version_ == 2) {
      OP_REQUIRES_OK(ctx,
                     ctx->input("feature_list_dense_missing_assumed_empty",
                                &feature_list_dense_missing_assumed_empty));
      OP_REQUIRES_OK(ctx,
                     ctx->input("context_dense_keys", &context_dense_keys));
      OP_REQUIRES_OK(ctx,
                     ctx->input("context_sparse_keys", &context_sparse_keys));
      OP_REQUIRES_OK(ctx,
                     ctx->input("context_ragged_keys", &context_ragged_keys));
      OP_REQUIRES_OK(
          ctx, ctx->input("feature_list_dense_keys", &feature_list_dense_keys));
      OP_REQUIRES_OK(ctx, ctx->input("feature_list_sparse_keys",
                                     &feature_list_sparse_keys));
      OP_REQUIRES_OK(ctx, ctx->input("feature_list_ragged_keys",
                                     &feature_list_ragged_keys));
      absl::call_once(flag_, [&]() {
        metrics::RecordParseDenseFeature(
            context_dense_keys->NumElements() +
            feature_list_dense_keys->NumElements());
        metrics::RecordParseSparseFeature(
            context_sparse_keys->NumElements() +
            feature_list_sparse_keys->NumElements());
        metrics::RecordParseRaggedFeature(
            context_ragged_keys->NumElements() +
            feature_list_ragged_keys->NumElements());
      });
    }

    // Validate input tensor shapes.
    OP_REQUIRES_OK(ctx, CheckInputShapes(
                            serialized, debug_name, context_dense_defaults,
                            context_dense_keys, context_sparse_keys,
                            context_ragged_keys, feature_list_dense_keys,
                            feature_list_sparse_keys, feature_list_ragged_keys,
                            feature_list_dense_missing_assumed_empty));

    example::FastParseExampleConfig context_config =
        MakeContextConfig(context_dense_keys, context_sparse_keys,
                          context_ragged_keys, context_dense_defaults);
    example::FastParseExampleConfig feature_list_config = MakeFeatureListConfig(
        feature_list_dense_keys, feature_list_sparse_keys,
        feature_list_ragged_keys, feature_list_dense_missing_assumed_empty);

    bool is_batch = TensorShapeUtils::IsVector(serialized->shape());
    auto serialized_t = serialized->flat<tstring>();
    auto debug_name_t = debug_name->flat<tstring>();
    gtl::ArraySlice<tstring> slice(serialized_t.data(), serialized_t.size());
    gtl::ArraySlice<tstring> names_slice(debug_name_t.data(),
                                         debug_name_t.size());

    example::Result context_result, feature_list_result;
    std::vector<Tensor> dense_feature_lengths;
    OP_REQUIRES_OK(
        ctx, FastParseSequenceExample(
                 context_config, feature_list_config, slice, names_slice,
                 ctx->device()->tensorflow_cpu_worker_threads()->workers,
                 &context_result, &feature_list_result, &dense_feature_lengths,
                 is_batch));

    OP_REQUIRES_OK(ctx, WriteOutput(context_result, feature_list_result,
                                    dense_feature_lengths, ctx));
  }

 protected:
  Status CheckInputShapes(
      const Tensor* serialized, const Tensor* names,
      const OpInputList& context_dense_defaults,

      const Tensor* context_dense_keys, const Tensor* context_sparse_keys,
      const Tensor* context_ragged_keys, const Tensor* feature_list_dense_keys,
      const Tensor* feature_list_sparse_keys,
      const Tensor* feature_list_ragged_keys,
      const Tensor* feature_list_dense_missing_assumed_empty) const {
    if (TensorShapeUtils::IsMatrixOrHigher(serialized->shape())) {
      return errors::InvalidArgument(
          "Expected serialized to be a scalar or vector, got shape: ",
          serialized->shape().DebugString());
    }
    if (op_version_ > 1) {
      if (context_dense_keys->NumElements() != attrs_.num_context_dense) {
        return errors::InvalidArgument(
            "Expected len(context_dense_keys) to match len(Tcontext_dense)");
      }
      if (context_sparse_keys->NumElements() != attrs_.num_context_sparse) {
        return errors::InvalidArgument(
            "Expected len(context_sparse_keys) to match Ncontext_sparse");
      }
      if (context_ragged_keys->NumElements() != attrs_.num_context_ragged) {
        return errors::InvalidArgument(
            "Expected len(context_ragged_keys) to match "
            "len(context_ragged_value_types)");
      }
      if (feature_list_dense_keys->NumElements() !=
          attrs_.num_feature_list_dense) {
        return errors::InvalidArgument(
            "Expected len(feature_list_dense_keys) to match "
            "Nfeature_list_dense");
      }
      if (feature_list_dense_missing_assumed_empty->NumElements() !=
          attrs_.num_feature_list_dense) {
        return errors::InvalidArgument(
            "Expected len(feature_list_dense_missing_assumed_empty to match "
            "Nfeature_list_dense");
      }
      if (feature_list_sparse_keys->NumElements() !=
          attrs_.num_feature_list_sparse) {
        return errors::InvalidArgument(
            "Expected len(feature_list_sparse_keys) to match "
            "Nfeature_list_sparse");
      }
      if (feature_list_ragged_keys->NumElements() !=
          attrs_.num_feature_list_ragged) {
        return errors::InvalidArgument(
            "Expected len(feature_list_ragged_keys) to match "
            "len(feature_list_ragged_value_types)");
      }
    }
    if (context_dense_defaults.size() != attrs_.num_context_dense) {
      return errors::InvalidArgument(
          "Expected len(context_dense_defaults) "
          "== len(context_dense_keys) but got: ",
          context_dense_defaults.size(), " vs. ", attrs_.num_context_dense);
    }
    for (int d = 0; d < attrs_.num_context_dense; ++d) {
      const Tensor& def_value = context_dense_defaults[d];
      if (def_value.NumElements() > 0) {
        if (def_value.shape() != attrs_.context_dense_shapes[d]) {
          return errors::InvalidArgument(
              "default_value[", d,
              "].shape() == ", def_value.shape().DebugString(),
              " != context_dense_shapes[", d,
              "] == ", attrs_.context_dense_shapes[d].DebugString());
        }
        if (def_value.dtype() != attrs_.context_dense_types[d]) {
          return errors::InvalidArgument(
              "context_dense_defaults[", d,
              "].dtype() == ", DataTypeString(def_value.dtype()),
              " != context_dense_types[", d,
              "] == ", DataTypeString(attrs_.context_dense_types[d]));
        }
      }
    }
    return Status::OK();
  }

  example::FastParseExampleConfig MakeContextConfig(
      const Tensor* dense_keys, const Tensor* sparse_keys,
      const Tensor* ragged_keys,
      const OpInputList& context_dense_defaults) const {
    example::FastParseExampleConfig config;
    for (int d = 0; d < attrs_.num_context_dense; ++d) {
      const tstring& key = dense_keys ? dense_keys->flat<tstring>()(d)
                                      : attrs_.context_dense_keys[d];
      config.dense.push_back({key, attrs_.context_dense_types[d],
                              attrs_.context_dense_shapes[d],
                              context_dense_defaults[d],
                              false /* attrs_.context_variable_length[d] */,
                              0 /*attrs_.context_elements_per_stride[d] */});
    }
    for (int d = 0; d < attrs_.num_context_sparse; ++d) {
      const tstring& key = sparse_keys ? sparse_keys->flat<tstring>()(d)
                                       : attrs_.context_sparse_keys[d];
      config.sparse.push_back({key, attrs_.context_sparse_types[d]});
    }
    for (int d = 0; d < attrs_.num_context_ragged; ++d) {
      config.ragged.push_back({ragged_keys->flat<tstring>()(d),
                               attrs_.context_ragged_value_types[d],
                               attrs_.context_ragged_split_types[d]});
    }
    return config;
  }

  example::FastParseExampleConfig MakeFeatureListConfig(
      const Tensor* dense_keys, const Tensor* sparse_keys,
      const Tensor* ragged_keys,
      const Tensor* feature_list_dense_missing_assumed_empty) const {
    example::FastParseExampleConfig config;
    for (int d = 0; d < attrs_.num_feature_list_dense; ++d) {
      const tstring& key = dense_keys ? dense_keys->flat<tstring>()(d)
                                      : attrs_.feature_list_dense_keys[d];
      bool missing_assumed_empty =
          feature_list_dense_missing_assumed_empty
              ? feature_list_dense_missing_assumed_empty->flat<bool>()(d)
              : attrs_.feature_list_dense_missing_assumed_empty.count(key) > 0;
      DataType dtype = attrs_.feature_list_dense_types[d];
      Tensor default_value = Tensor(dtype, TensorShape({}));
      config.dense.push_back(
          {key, dtype, attrs_.feature_list_dense_shapes[d], default_value,
           missing_assumed_empty,
           0 /*attrs_.feature_list_elements_per_stride[d] */});
    }
    for (int d = 0; d < attrs_.num_feature_list_sparse; ++d) {
      const tstring& key = sparse_keys ? sparse_keys->flat<tstring>()(d)
                                       : attrs_.feature_list_sparse_keys[d];
      config.sparse.push_back({key, attrs_.feature_list_sparse_types[d]});
    }
    for (int d = 0; d < attrs_.num_feature_list_ragged; ++d) {
      config.ragged.push_back({ragged_keys->flat<tstring>()(d),
                               attrs_.feature_list_ragged_value_types[d],
                               attrs_.feature_list_ragged_split_types[d]});
    }
    return config;
  }

  Status WriteOutput(const example::Result& context_result,
                     const example::Result& feature_list_result,
                     const std::vector<Tensor>& dense_feature_lengths,
                     OpKernelContext* ctx) const {
    OpOutputList context_sparse_indices;
    OpOutputList context_sparse_values;
    OpOutputList context_sparse_shapes;
    OpOutputList context_dense_values;
    OpOutputList feature_list_sparse_indices;
    OpOutputList feature_list_sparse_values;
    OpOutputList feature_list_sparse_shapes;
    OpOutputList feature_list_dense_values;
    OpOutputList feature_list_dense_lengths;

    TF_RETURN_IF_ERROR(
        ctx->output_list("context_sparse_indices", &context_sparse_indices));
    TF_RETURN_IF_ERROR(
        ctx->output_list("context_sparse_values", &context_sparse_values));
    TF_RETURN_IF_ERROR(
        ctx->output_list("context_sparse_shapes", &context_sparse_shapes));
    TF_RETURN_IF_ERROR(
        ctx->output_list("context_dense_values", &context_dense_values));
    TF_RETURN_IF_ERROR(
        ctx->output_list("context_sparse_indices", &context_sparse_indices));
    TF_RETURN_IF_ERROR(ctx->output_list("feature_list_sparse_indices",
                                        &feature_list_sparse_indices));
    TF_RETURN_IF_ERROR(ctx->output_list("feature_list_sparse_values",
                                        &feature_list_sparse_values));
    TF_RETURN_IF_ERROR(ctx->output_list("feature_list_sparse_shapes",
                                        &feature_list_sparse_shapes));
    TF_RETURN_IF_ERROR(ctx->output_list("feature_list_dense_values",
                                        &feature_list_dense_values));
    TF_RETURN_IF_ERROR(ctx->output_list("feature_list_dense_lengths",
                                        &feature_list_dense_lengths));
    for (int d = 0; d < attrs_.num_context_dense; ++d) {
      context_dense_values.set(d, context_result.dense_values[d]);
    }
    for (int d = 0; d < attrs_.num_feature_list_dense; ++d) {
      feature_list_dense_values.set(d, feature_list_result.dense_values[d]);
      feature_list_dense_lengths.set(d, dense_feature_lengths[d]);
    }
    for (int d = 0; d < attrs_.num_context_sparse; ++d) {
      context_sparse_indices.set(d, context_result.sparse_indices[d]);
      context_sparse_values.set(d, context_result.sparse_values[d]);
      context_sparse_shapes.set(d, context_result.sparse_shapes[d]);
    }
    for (int d = 0; d < attrs_.num_feature_list_sparse; ++d) {
      feature_list_sparse_indices.set(d, feature_list_result.sparse_indices[d]);
      feature_list_sparse_values.set(d, feature_list_result.sparse_values[d]);
      feature_list_sparse_shapes.set(d, feature_list_result.sparse_shapes[d]);
    }
    if (op_version_ == 2) {
      OpOutputList context_ragged_values;
      OpOutputList context_ragged_splits;
      OpOutputList feature_list_ragged_values;
      OpOutputList feature_list_ragged_inner_splits;
      OpOutputList feature_list_ragged_outer_splits;
      TF_RETURN_IF_ERROR(
          ctx->output_list("context_ragged_values", &context_ragged_values));
      TF_RETURN_IF_ERROR(ctx->output_list("context_ragged_row_splits",
                                          &context_ragged_splits));
      TF_RETURN_IF_ERROR(ctx->output_list("feature_list_ragged_values",
                                          &feature_list_ragged_values));
      TF_RETURN_IF_ERROR(ctx->output_list("feature_list_ragged_inner_splits",
                                          &feature_list_ragged_inner_splits));
      TF_RETURN_IF_ERROR(ctx->output_list("feature_list_ragged_outer_splits",
                                          &feature_list_ragged_outer_splits));
      for (int d = 0; d < attrs_.num_context_ragged; ++d) {
        context_ragged_values.set(d, context_result.ragged_values[d]);
        context_ragged_splits.set(d, context_result.ragged_splits[d]);
      }
      for (int d = 0; d < attrs_.num_feature_list_ragged; ++d) {
        feature_list_ragged_values.set(d, feature_list_result.ragged_values[d]);
        feature_list_ragged_outer_splits.set(
            d, feature_list_result.ragged_outer_splits[d]);
        feature_list_ragged_inner_splits.set(
            d, feature_list_result.ragged_splits[d]);
      }
    }
    return Status::OK();
  }

  ParseSequenceExampleAttrs attrs_;
  int op_version_;
  absl::once_flag flag_;
};

REGISTER_KERNEL_BUILDER(Name("ParseSequenceExample").Device(DEVICE_CPU),
                        ParseSequenceExampleOp);
REGISTER_KERNEL_BUILDER(Name("ParseSequenceExampleV2").Device(DEVICE_CPU),
                        ParseSequenceExampleOp);

class ParseSingleSequenceExampleOp : public OpKernel {
 public:
  explicit ParseSingleSequenceExampleOp(OpKernelConstruction* ctx)
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
    absl::call_once(
        flag_, [&context_dense_keys_t, &context_sparse_keys_t,
                &feature_list_dense_keys_t, &feature_list_sparse_keys_t]() {
          metrics::RecordParseDenseFeature(context_dense_keys_t.size() +
                                           feature_list_dense_keys_t.size());
          metrics::RecordParseSparseFeature(context_sparse_keys_t.size() +
                                            feature_list_sparse_keys_t.size());
        });
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
                      "] to be a scalar, got shape: ",
                      context_dense_keys[di].shape().DebugString()));
      context_dense_keys_t[di] = context_dense_keys[di].scalar<tstring>()();
    }
    for (int di = 0; di < attrs_.num_context_sparse; ++di) {
      OP_REQUIRES(ctx,
                  TensorShapeUtils::IsScalar(context_sparse_keys[di].shape()),
                  errors::InvalidArgument(
                      "Expected context_sparse_keys[", di,
                      "] to be a scalar, got shape: ",
                      context_sparse_keys[di].shape().DebugString()));
      context_sparse_keys_t[di] = context_sparse_keys[di].scalar<tstring>()();
    }
    for (int di = 0; di < attrs_.num_feature_list_dense; ++di) {
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(feature_list_dense_keys[di].shape()),
          errors::InvalidArgument(
              "Expected feature_list_dense_keys[", di,
              "] to be a scalar, got shape: ",
              feature_list_dense_keys[di].shape().DebugString()));
      feature_list_dense_keys_t[di] =
          feature_list_dense_keys[di].scalar<tstring>()();
    }
    for (int di = 0; di < attrs_.num_feature_list_sparse; ++di) {
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(feature_list_sparse_keys[di].shape()),
          errors::InvalidArgument(
              "Expected feature_list_sparse_keys[", di,
              "] to be a scalar, got shape: ",
              feature_list_sparse_keys[di].shape().DebugString()));
      feature_list_sparse_keys_t[di] =
          feature_list_sparse_keys[di].scalar<tstring>()();
    }
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsVector(
            feature_list_dense_missing_assumed_empty->shape()),
        errors::InvalidArgument(
            "Expected feature_list_dense_missing_assumed_empty ",
            "to be a vector, got shape: ",
            feature_list_dense_missing_assumed_empty->shape().DebugString()));
    auto feature_list_dense_missing_assumped_empty_t =
        feature_list_dense_missing_assumed_empty->vec<tstring>();
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
    auto debug_name_t = debug_name->scalar<tstring>();

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
        OP_REQUIRES(ctx, def_value.shape() == attrs_.context_dense_shapes[d],
                    errors::InvalidArgument(
                        "def_value[", d,
                        "].shape() == ", def_value.shape().DebugString(),
                        " != context_dense_shapes_[", d,
                        "] == ", attrs_.context_dense_shapes[d].DebugString()));
        OP_REQUIRES(
            ctx, def_value.dtype() == attrs_.context_dense_types[d],
            errors::InvalidArgument(
                "context_dense_defaults[", d, "].dtype() == ",
                DataTypeString(def_value.dtype()), " != context_dense_types_[",
                d, "] == ", DataTypeString(attrs_.context_dense_types[d])));
      }
    }

    auto serialized_t = serialized->scalar<tstring>();

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

#ifdef TENSORFLOW_LITE_PROTOS
    SequenceExample ex;
#else
    // Allocate the SequenceExample on an arena. Provides better memory locality
    // and greatly speeds up destruction.
    protobuf::ArenaOptions options;
    // We have some hint of what the final proto size will be based on the size
    // of the serialized bytes- use this to set a custom allocation strategy.
    // Note that the default allocation strategy is quite conservative (min
    // block size of 256 bytes, and a max of 8 kilobytes).
    const size_t block_size = serialized_t().size() * 1.1;
    options.start_block_size = std::max(options.start_block_size, block_size);
    options.max_block_size = std::max(options.max_block_size, block_size);
    protobuf::Arena arena(options);
    auto& ex = *protobuf::Arena::CreateMessage<SequenceExample>(&arena);
#endif
    OP_REQUIRES(
        ctx, ParseProtoUnlimited(&ex, serialized_t()),
        errors::InvalidArgument("Could not parse example input, value: '",
                                serialized_t(), "'"));

    const tstring& name = (has_debug_name) ? debug_name_t() : "<unknown>";
    const Features& context = ex.context();
    const auto& context_dict = context.feature();

    // Context Dense -----------------------------------------------------------

    // Preallocate context_dense_values, since we know their sizes
    for (int d = 0; d < attrs_.num_context_dense; ++d) {
      TensorShape out_shape;
      for (const int dim : attrs_.context_dense_shapes[d].dim_sizes())
        out_shape.AddDim(dim);
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, context_dense_values.allocate(d, out_shape, &out));
    }

    for (int d = 0; d < attrs_.num_context_dense; ++d) {
      const tstring& key = context_dense_keys_t[d];
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
                                    "  Feature is: ", f.DebugString()));

        OP_REQUIRES_OK(ctx, FeatureDenseCopy(0, name, key, dtype, shape, f,
                                             context_dense_values[d]));
      } else {
        RowDenseCopy(0, dtype, context_dense_defaults[d],
                     context_dense_values[d]);
      }
    }

    // Context Sparse ----------------------------------------------------------
    for (int d = 0; d < attrs_.num_context_sparse; ++d) {
      const tstring& key = context_sparse_keys_t[d];
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
                                    "  Feature is: ", f.DebugString()));

        Tensor feature_values = FeatureSparseCopy(0, key, dtype, f);
        const int64 num_elements = feature_values.NumElements();
        TensorShape indices_shape({num_elements, 1});
        Tensor* sp_indices_d = nullptr;
        Tensor* sp_shape_d = nullptr;
        OP_REQUIRES_OK(ctx, context_sparse_indices.allocate(d, indices_shape,
                                                            &sp_indices_d));
        context_sparse_values.set(d, feature_values);
        OP_REQUIRES_OK(ctx, context_sparse_shapes.allocate(d, TensorShape({1}),
                                                           &sp_shape_d));
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
        OP_REQUIRES_OK(ctx, context_sparse_indices.allocate(d, indices_shape,
                                                            &sp_indices_d));
        OP_REQUIRES_OK(
            ctx, context_sparse_values.allocate(d, values_shape, &sp_values_d));
        OP_REQUIRES_OK(ctx, context_sparse_shapes.allocate(d, TensorShape({1}),
                                                           &sp_shape_d));
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
      const tstring& key = feature_list_dense_keys_t[d];
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
      OP_REQUIRES_OK(ctx,
                     feature_list_dense_values.allocate(d, out_shape, &out));

      for (int64 t = 0; t < fl.feature_size(); ++t) {
        const Feature& f = fl.feature(t);
        bool types_match;
        OP_REQUIRES_OK(ctx, CheckTypesMatch(f, dtype, &types_match));
        OP_REQUIRES(ctx, types_match,
                    errors::InvalidArgument(
                        "Name: ", name, ", Feature list: ", key, ", Index: ", t,
                        ".  Data types don't match. ",
                        "Expected type: ", DataTypeString(dtype),
                        "  Feature is: ", f.DebugString()));
        OP_REQUIRES_OK(ctx, FeatureDenseCopy(t, name, key, dtype, shape, f,
                                             feature_list_dense_values[d]));
      }
    }

    // Feature List Sparse -----------------------------------------------------
    for (int d = 0; d < attrs_.num_feature_list_sparse; ++d) {
      const tstring& key = feature_list_sparse_keys_t[d];
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
                                      "  Feature is: ", f.DebugString()));
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
      OP_REQUIRES_OK(ctx, feature_list_sparse_indices.allocate(d, indices_shape,
                                                               &sp_indices_d));
      OP_REQUIRES_OK(ctx, feature_list_sparse_values.allocate(d, values_shape,
                                                              &sp_values_d));
      OP_REQUIRES_OK(ctx, feature_list_sparse_shapes.allocate(
                              d, TensorShape({2}), &sp_shape_d));
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
  absl::once_flag flag_;
};

REGISTER_KERNEL_BUILDER(Name("ParseSingleSequenceExample").Device(DEVICE_CPU),
                        ParseSingleSequenceExampleOp);

#ifndef IS_MOBILE_PLATFORM
// when using lite protos on mobile, decoding JSON is not available.

class DecodeJSONExampleOp : public OpKernel {
 public:
  explicit DecodeJSONExampleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    resolver_.reset(protobuf::util::NewTypeResolverForDescriptorPool(
        "type.googleapis.com", protobuf::DescriptorPool::generated_pool()));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* json_examples;
    OP_REQUIRES_OK(ctx, ctx->input("json_examples", &json_examples));
    Tensor* binary_examples;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("binary_examples", json_examples->shape(),
                                  &binary_examples));

    for (int i = 0; i < json_examples->NumElements(); ++i) {
      const tstring& json_example = json_examples->flat<tstring>()(i);
      protobuf::io::ArrayInputStream in(json_example.data(),
                                        json_example.size());
      TStringOutputStream out(&binary_examples->flat<tstring>()(i));
      auto status = protobuf::util::JsonToBinaryStream(
          resolver_.get(), "type.googleapis.com/tensorflow.Example", &in, &out);
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
