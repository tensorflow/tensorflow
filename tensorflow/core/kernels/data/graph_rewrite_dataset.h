/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_GRAPH_REWRITE_DATASET_H_
#define TENSORFLOW_CORE_KERNELS_DATA_GRAPH_REWRITE_DATASET_H_

#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"

namespace tensorflow {
namespace data {

class GraphRewriteDataset : public DatasetBase {
 public:
  GraphRewriteDataset(OpKernelContext* ctx, const DatasetBase* input,
                      const DataTypeVector& output_types,
                      const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        optimized_input_(nullptr),
        input_(input),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
  }

  ~GraphRewriteDataset() override;

  // Runs Grappler to transform the input dataset into optimized_input_
  // dataset.
  Status Optimize(OpKernelContext* ctx);

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override;

 private:
  class Iterator;

  // Create a Grappler RewriteConfig proto that defines the list of
  // optimizations to be run by the Grappler Meta Optimizer.
  virtual RewriterConfig CreateGrapplerRewriteConfig() = 0;

  Status ApplyOptimizations(OpKernelContext* ctx, GraphDef* graph_def,
                            string* output_node);

  DatasetBase* optimized_input_;
  FunctionLibraryRuntime* lib_ = nullptr;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_ = nullptr;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_ = nullptr;
  std::unique_ptr<FunctionHandleCache> function_handle_cache_ = nullptr;
  const DatasetBase* input_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_GRAPH_REWRITE_DATASET_H_
