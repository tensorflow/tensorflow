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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/data/dataset.h"

namespace tensorflow {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.
class DatasetToGraphOp : public OpKernel {
 public:
  explicit DatasetToGraphOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
    GraphDefBuilder b;
    DatasetBase::DatasetGraphDefBuilder db(&b);
    Node* input_node = nullptr;
    OP_REQUIRES_OK(ctx, db.AddParentDataset(ctx, dataset, &input_node));
    GraphDef graph_def;
    OP_REQUIRES_OK(ctx, b.ToGraphDef(&graph_def));
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &result));
    result->scalar<string>()() = graph_def.SerializeAsString();
  }
};

REGISTER_KERNEL_BUILDER(Name("DatasetToGraph").Device(DEVICE_CPU),
                        DatasetToGraphOp);

}  // namespace tensorflow
