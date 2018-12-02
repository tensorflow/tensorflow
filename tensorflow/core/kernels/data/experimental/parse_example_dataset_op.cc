/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <deque>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/kernels/data/parallel_map_iterator.h"
#include "tensorflow/core/util/example_proto_fast_parsing.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
class ParseExampleDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParseExampleDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_keys", &sparse_keys_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_keys", &dense_keys_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_types", &sparse_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tdense", &dense_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_shapes", &dense_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sloppy", &sloppy_));
    for (int i = 0; i < dense_shapes_.size(); ++i) {
      bool shape_ok = true;
      if (dense_shapes_[i].dims() == -1) {
        shape_ok = false;
      } else {
        for (int d = 1; d < dense_shapes_[i].dims(); ++d) {
          if (dense_shapes_[i].dim_size(d) == -1) {
            shape_ok = false;
          }
        }
      }
      OP_REQUIRES(ctx, shape_ok,
                  errors::InvalidArgument(
                      "dense_shapes[", i,
                      "] has unknown rank or unknown inner dimensions: ",
                      dense_shapes_[i].DebugString()));
      TensorShape dense_shape;
      if (dense_shapes_[i].dims() > 0 && dense_shapes_[i].dim_size(0) == -1) {
        variable_length_.push_back(true);
        for (int d = 1; d < dense_shapes_[i].dims(); ++d) {
          dense_shape.AddDim(dense_shapes_[i].dim_size(d));
        }
      } else {
        variable_length_.push_back(false);
        dense_shapes_[i].AsTensorShape(&dense_shape);
      }
      elements_per_stride_.push_back(dense_shape.num_elements());
    }
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 num_parallel_calls;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                            &num_parallel_calls));
    OP_REQUIRES(ctx, num_parallel_calls > 0,
                errors::InvalidArgument(
                    "num_parallel_calls must be greater than zero."));

    OpInputList dense_default_tensors;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("dense_defaults", &dense_default_tensors));

    OP_REQUIRES(ctx, dense_default_tensors.size() == dense_keys_.size(),
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_default_tensors.size(), " vs. ", dense_keys_.size()));

    std::vector<Tensor> dense_defaults(dense_default_tensors.begin(),
                                       dense_default_tensors.end());

    for (int d = 0; d < dense_keys_.size(); ++d) {
      const Tensor& def_value = dense_defaults[d];
      if (variable_length_[d]) {
        OP_REQUIRES(ctx, def_value.NumElements() == 1,
                    errors::InvalidArgument(
                        "dense_shape[", d, "] is a variable length shape: ",
                        dense_shapes_[d].DebugString(),
                        ", therefore "
                        "def_value[",
                        d,
                        "] must contain a single element ("
                        "the padding element).  But its shape is: ",
                        def_value.shape().DebugString()));
      } else if (def_value.NumElements() > 0) {
        OP_REQUIRES(ctx, dense_shapes_[d].IsCompatibleWith(def_value.shape()),
                    errors::InvalidArgument(
                        "def_value[", d,
                        "].shape() == ", def_value.shape().DebugString(),
                        " is not compatible with dense_shapes_[", d,
                        "] == ", dense_shapes_[d].DebugString()));
      }
      OP_REQUIRES(ctx, def_value.dtype() == dense_types_[d],
                  errors::InvalidArgument(
                      "dense_defaults[", d, "].dtype() == ",
                      DataTypeString(def_value.dtype()), " != dense_types_[", d,
                      "] == ", DataTypeString(dense_types_[d])));
    }

    example::FastParseExampleConfig config;
    std::map<string, int> key_to_output_index;
    for (int d = 0; d < dense_keys_.size(); ++d) {
      config.dense.push_back({dense_keys_[d], dense_types_[d], dense_shapes_[d],
                              dense_default_tensors[d], variable_length_[d],
                              elements_per_stride_[d]});
      auto result = key_to_output_index.insert({dense_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          dense_keys_[d]));
    }
    for (int d = 0; d < sparse_keys_.size(); ++d) {
      config.sparse.push_back({sparse_keys_[d], sparse_types_[d]});
      auto result = key_to_output_index.insert({sparse_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          sparse_keys_[d]));
    }
    int i = 0;
    for (auto it = key_to_output_index.begin(); it != key_to_output_index.end();
         it++) {
      it->second = i++;
    }

    *output =
        new Dataset(ctx, input, dense_defaults, sparse_keys_, dense_keys_,
                    std::move(key_to_output_index), std::move(config),
                    num_parallel_calls, sparse_types_, dense_types_,
                    dense_shapes_, output_types_, output_shapes_, sloppy_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::vector<Tensor> dense_defaults, std::vector<string> sparse_keys,
            std::vector<string> dense_keys,
            std::map<string, int> key_to_output_index,
            example::FastParseExampleConfig config, int32 num_parallel_calls,
            const DataTypeVector& sparse_types,
            const DataTypeVector& dense_types,
            const std::vector<PartialTensorShape>& dense_shapes,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes, bool sloppy)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          dense_defaults_(std::move(dense_defaults)),
          sparse_keys_(std::move(sparse_keys)),
          dense_keys_(std::move(dense_keys)),
          key_to_output_index_(std::move(key_to_output_index)),
          config_(std::move(config)),
          num_parallel_calls_(num_parallel_calls),
          sparse_types_(sparse_types),
          dense_types_(dense_types),
          dense_shapes_(dense_shapes),
          output_types_(output_types),
          output_shapes_(output_shapes),
          sloppy_(sloppy) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      std::unique_ptr<ParallelMapFunctor> parse_example_functor(
          new ParseExampleFunctor(this));
      return NewParallelMapIterator(
          {this, strings::StrCat(prefix, "::ParseExample")}, input_,
          std::move(parse_example_functor), num_parallel_calls_, sloppy_);
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "ParseExampleDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      Node* num_parallle_calls_node;
      std::vector<Node*> dense_defaults_nodes;
      dense_defaults_nodes.reserve(dense_defaults_.size());

      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallle_calls_node));

      for (const Tensor& dense_default : dense_defaults_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(dense_default, &node));
        dense_defaults_nodes.emplace_back(node);
      }

      AttrValue sparse_keys_attr;
      AttrValue dense_keys_attr;
      AttrValue sparse_types_attr;
      AttrValue dense_attr;
      AttrValue dense_shapes_attr;
      AttrValue sloppy_attr;

      b->BuildAttrValue(sparse_keys_, &sparse_keys_attr);
      b->BuildAttrValue(dense_keys_, &dense_keys_attr);
      b->BuildAttrValue(sparse_types_, &sparse_types_attr);
      b->BuildAttrValue(dense_types_, &dense_attr);
      b->BuildAttrValue(dense_shapes_, &dense_shapes_attr);
      b->BuildAttrValue(sloppy_, &sloppy_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(this,
                                       {
                                           {0, input_graph_node},
                                           {1, num_parallle_calls_node},
                                       },
                                       {{2, dense_defaults_nodes}},
                                       {{"sparse_keys", sparse_keys_attr},
                                        {"dense_keys", dense_keys_attr},
                                        {"sparse_types", sparse_types_attr},
                                        {"Tdense", dense_attr},
                                        {"dense_shapes", dense_shapes_attr},
                                        {"sloppy", sloppy_attr}},
                                       output));
      return Status::OK();
    }

   private:
    class ParseExampleFunctor : public ParallelMapFunctor {
     public:
      explicit ParseExampleFunctor(const Dataset* dataset)
          : dataset_(dataset) {}

      void MapFunc(IteratorContext* ctx, const string& prefix,
                   std::vector<Tensor> input, std::vector<Tensor>* output,
                   StatusCallback callback) override {
        (*ctx->runner())([this, ctx, input, output, callback]() {
          thread::ThreadPool* device_threadpool =
              ctx->lib()->device()->tensorflow_cpu_worker_threads()->workers;
          std::vector<string> slice_vec;
          for (const Tensor& t : input) {
            auto serialized_t = t.flat<string>();
            gtl::ArraySlice<string> slice(serialized_t.data(),
                                          serialized_t.size());
            for (auto it = slice.begin(); it != slice.end(); it++)
              slice_vec.push_back(*it);
          }
          example::FastParseExampleConfig config = dataset_->config_;
          // local copy of config_ for modification.
          auto stats_aggregator = ctx->stats_aggregator();
          if (stats_aggregator) {
            config.collect_feature_stats = true;
          }
          example::Result example_result;
          Status s = FastParseExample(config, slice_vec, {}, device_threadpool,
                                      &example_result);
          if (s.ok()) {
            (*output).resize(dataset_->key_to_output_index_.size());
            for (int d = 0; d < dataset_->dense_keys_.size(); ++d) {
              int output_index =
                  dataset_->key_to_output_index_.at(dataset_->dense_keys_[d]);
              DCHECK(example_result.dense_values[d].dtype() ==
                     dataset_->output_dtypes()[output_index])
                  << "Got wrong type for FastParseExample return value " << d
                  << " (expected "
                  << DataTypeString(dataset_->output_dtypes()[output_index])
                  << ", got "
                  << DataTypeString(example_result.dense_values[d].dtype())
                  << ").";
              DCHECK(dataset_->output_shapes()[output_index].IsCompatibleWith(
                  example_result.dense_values[d].shape()))
                  << "Got wrong shape for FastParseExample return value " << d
                  << " (expected "
                  << dataset_->output_shapes()[output_index].DebugString()
                  << ", got "
                  << example_result.dense_values[d].shape().DebugString()
                  << ").";
              (*output)[output_index] = example_result.dense_values[d];
            }
            for (int d = 0; d < dataset_->sparse_keys_.size(); ++d) {
              int output_index =
                  dataset_->key_to_output_index_.at(dataset_->sparse_keys_[d]);
              (*output)[output_index] =
                  Tensor(ctx->allocator({}), DT_VARIANT, {3});
              Tensor& serialized_sparse = (*output)[output_index];
              auto serialized_sparse_t = serialized_sparse.vec<Variant>();
              serialized_sparse_t(0) = example_result.sparse_indices[d];
              serialized_sparse_t(1) = example_result.sparse_values[d];
              serialized_sparse_t(2) = example_result.sparse_shapes[d];
              DCHECK(serialized_sparse.dtype() ==
                     dataset_->output_dtypes()[output_index])
                  << "Got wrong type for FastParseExample return value " << d
                  << " (expected "
                  << DataTypeString(dataset_->output_dtypes()[output_index])
                  << ", got " << DataTypeString(serialized_sparse.dtype())
                  << ").";
              DCHECK(dataset_->output_shapes()[output_index].IsCompatibleWith(
                  serialized_sparse.shape()))
                  << "Got wrong shape for FastParseExample return value " << d
                  << " (expected "
                  << dataset_->output_shapes()[output_index].DebugString()
                  << ", got " << serialized_sparse.shape().DebugString()
                  << ").";
            }
            // TODO(b/111553342): User provided tags instead of fixed tag.
            if (stats_aggregator) {
              stats_aggregator->IncrementCounter(
                  "examples_count", "trainer",
                  example_result.feature_stats.size());
              for (example::PerExampleFeatureStats feature_stats :
                   example_result.feature_stats) {
                stats_aggregator->AddToHistogram(
                    "features",
                    {static_cast<double>(feature_stats.features_count)});
                stats_aggregator->IncrementCounter(
                    "features_count", "trainer", feature_stats.features_count);
                stats_aggregator->IncrementCounter(
                    "feature_values_count", "trainer",
                    feature_stats.feature_values_count);
                stats_aggregator->AddToHistogram(
                    "feature-values",
                    {static_cast<double>(feature_stats.feature_values_count)});
              }
            }
          }
          callback(s);
        });
      }

     private:
      const Dataset* dataset_;
    };

    const DatasetBase* const input_;
    const std::vector<Tensor> dense_defaults_;
    const std::vector<string> sparse_keys_;
    const std::vector<string> dense_keys_;
    const std::map<string, int> key_to_output_index_;
    const example::FastParseExampleConfig config_;
    const int64 num_parallel_calls_;
    const DataTypeVector sparse_types_;
    const DataTypeVector dense_types_;
    const std::vector<PartialTensorShape> dense_shapes_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const bool sloppy_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  bool sloppy_;
  std::vector<string> sparse_keys_;
  std::vector<string> dense_keys_;
  DataTypeVector sparse_types_;
  DataTypeVector dense_types_;
  std::vector<PartialTensorShape> dense_shapes_;
  std::vector<bool> variable_length_;
  std::vector<std::size_t> elements_per_stride_;
};

REGISTER_KERNEL_BUILDER(
    Name("ExperimentalParseExampleDataset").Device(DEVICE_CPU),
    ParseExampleDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
