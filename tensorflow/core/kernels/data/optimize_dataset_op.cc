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
#include <map>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
class OptimizeDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit OptimizeDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::vector<string> optimizations;
    OP_REQUIRES_OK(
        ctx, ParseVectorArgument<string>(ctx, "optimizations", &optimizations));
    Dataset* dataset =
        new Dataset(ctx, input, optimizations, output_types_, output_shapes_);
    Status s = dataset->Optimize(ctx);
    if (s.ok()) {
      *output = dataset;
    } else {
      dataset->Unref();
      OP_REQUIRES_OK(ctx, s);
    }
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const std::vector<string>& optimizations,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          optimized_input_(nullptr),
          input_(input),
          optimizations_(optimizations),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override {
      input_->Unref();
      if (optimized_input_) {
        optimized_input_->Unref();
      }
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      // We do not add a token for the optimization dataset to the prefix. The
      // prefix is used to identify checkpoint elements and since the
      // optimization dataset is excluded from the checkpoint, adding a token
      // here would result in invalid checkpoint identifiers.
      return std::unique_ptr<IteratorBase>(new Iterator({this, prefix}));
    }

    Status Optimize(OpKernelContext* ctx) {
      GraphDefBuilder b;
      DatasetGraphDefBuilder db(&b);
      Node* input_node = nullptr;
      SerializationContext::Params params;
      std::vector<std::pair<string, Tensor>> input_list;
      params.flib_def = ctx->function_library()->GetFunctionLibraryDefinition();
      params.input_list = &input_list;
      params.optimization_only = true;
      SerializationContext serialization_ctx(params);
      TF_RETURN_IF_ERROR(
          db.AddInputDataset(&serialization_ctx, input_, &input_node));
      string output_node = input_node->name();

      GraphDef graph_def;
      TF_RETURN_IF_ERROR(b.ToGraphDef(&graph_def));
      VLOG(3) << "Before optimization: " << graph_def.DebugString();

      TF_RETURN_IF_ERROR(ApplyOptimizations(ctx, &graph_def, &output_node));
      VLOG(3) << "After optimization: " << graph_def.DebugString();

      // Instantiate the optimized input pipeline by running the optimized graph
      // using the optimized function library.
      TF_RETURN_IF_ERROR(
          ctx->function_library()->Clone(&flib_def_, &pflr_, &lib_));

      // Create a FunctionHandleCache.
      function_handle_cache_.reset(new FunctionHandleCache(lib_));

      // Some functions may have been modified without having their names
      // changed (for example, nested dataset graphs from FlatMap or
      // Interleave). To avoid name conflicts, we remove these functions from
      // flib_def_ before adding the optimized function library.
      for (const FunctionDef& fd : graph_def.library().function()) {
        if (flib_def_->Find(fd.signature().name()) != nullptr) {
          TF_RETURN_IF_ERROR(flib_def_->RemoveFunction(fd.signature().name()));
        }
      }
      TF_RETURN_IF_ERROR(flib_def_->AddLibrary(graph_def.library()));

      Graph graph(OpRegistry::Global());
      TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));
      std::vector<Tensor> outputs;
      GraphRunner graph_runner(ctx->function_library()->device());

      TF_RETURN_IF_ERROR(
          graph_runner.Run(&graph, lib_, input_list, {output_node}, &outputs));
      TF_RETURN_IF_ERROR(
          GetDatasetFromVariantTensor(outputs[0], &optimized_input_));
      optimized_input_->Ref();
      return Status::OK();
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "OptimizeDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      // We only serialize the optimized dataset to avoid re-running
      // optimizations when the input pipeline is restored from a checkpoint.
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, optimized_input_, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        IteratorContext::Params params(ctx);
        params.lib = dataset()->lib_;
        params.function_handle_cache = dataset()->function_handle_cache_.get();
        return dataset()->optimized_input_->MakeIterator(
            IteratorContext(std::move(params)), prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        IteratorContext::Params params(ctx);
        params.lib = dataset()->lib_;
        params.function_handle_cache = dataset()->function_handle_cache_.get();
        return input_impl_->GetNext(IteratorContext(std::move(params)),
                                    out_tensors, end_of_sequence);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      std::unique_ptr<IteratorBase> input_impl_;
    };

    Status ApplyOptimizations(OpKernelContext* ctx, GraphDef* graph_def,
                              string* output_node) {
      // Add an identity node as the fetch node, otherwise we might get
      // 'placeholder is both fed and fetched' errors in some cases when using
      // input list with placeholder dataset nodes.
      NodeDef* node = graph_def->mutable_node()->Add();
      tensorflow::grappler::graph_utils::SetUniqueGraphNodeName(
          "Sink", graph_def, node);
      node->set_op("Identity");
      node->add_input(*output_node);
      (*node->mutable_attr())["T"].set_type(DT_VARIANT);
      *output_node = node->name();

      // Create metagraph.
      MetaGraphDef meta_graph_def;
      (*meta_graph_def.mutable_graph_def()) = *graph_def;

      // Grappler determines fetch ops from collection 'train_op'.
      CollectionDef collection_def;
      auto node_list = collection_def.mutable_node_list();
      node_list->add_value(*output_node);
      (*meta_graph_def.mutable_collection_def())["train_op"] = collection_def;

      // Create Grappler item.
      tensorflow::ConfigProto config;
      RewriterConfig& rewriter_config =
          *config.mutable_graph_options()->mutable_rewrite_options();
      for (const string& optimization : optimizations_) {
        rewriter_config.add_optimizers(optimization);
      }
      // If no optimizations were specified, supply a non-existent
      // optimization to prevent Grappler from applying the default set of
      // optimizations as some of them do not work out of the box at the
      // moment (e.g. because we have no cost model for dataset ops).
      if (optimizations_.empty()) {
        rewriter_config.add_optimizers("non-existent");
      } else {
        // If we apply custom dataset optimizers, explicitly trigger a subset of
        // standard grappler optimizations to further optimize modified dataset
        // graphs (e.g. performing constant folding on merged functions,
        // removing unused graph nodes)
        // TODO(b/118175421): This should be part of the tf.data optimization
        // pass manager.
        for (const auto& optimizer : {"pruning", "function", "constfold",
                                      "shape", "arithmetic", "dependency"}) {
          rewriter_config.add_optimizers(optimizer);
        }
      }
      tensorflow::grappler::ItemConfig item_config;
      item_config.apply_optimizations = true;
      std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
          tensorflow::grappler::GrapplerItemFromMetaGraphDef(
              "graph", meta_graph_def, item_config);
      std::unordered_map<string, tensorflow::DeviceProperties> device_map;
      tensorflow::grappler::VirtualCluster cluster(device_map);

      // Run optimizer.
      if (VLOG_IS_ON(2)) {
        LOG(INFO) << "Performing the following optimizations:";
        for (const string& optimization : optimizations_) {
          LOG(INFO) << "  " << optimization;
        }
      }
      TF_RETURN_IF_ERROR(tensorflow::grappler::RunMetaOptimizer(
          *grappler_item, config, ctx->device(), &cluster, graph_def));

      return Status::OK();
    }

    DatasetBase* optimized_input_;
    FunctionLibraryRuntime* lib_ = nullptr;
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_ = nullptr;
    std::unique_ptr<FunctionLibraryDefinition> flib_def_ = nullptr;
    std::unique_ptr<FunctionHandleCache> function_handle_cache_ = nullptr;
    const DatasetBase* input_;
    const std::vector<string> optimizations_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("OptimizeDataset").Device(DEVICE_CPU),
                        OptimizeDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
