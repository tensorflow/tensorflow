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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/tensor_forest/resources.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

class TensorForestTreePredictOp : public OpKernel {
 public:
  explicit TensorForestTreePredictOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("logits_dimension", &logits_dimension_));
  }

  void Compute(OpKernelContext* context) override {
    TensorForestTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const Tensor* dense_features = nullptr;
    OP_REQUIRES_OK(context, context->input("dense_features", &dense_features));

    auto data_set = dense_features->matrix<float>();
    const int32 batch_size = dense_features->dim_size(0);

    Tensor* output_predictions = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {batch_size, logits_dimension_},
                                            &output_predictions));
    auto out = output_predictions->matrix<float>();

    if (decision_tree_resource->get_size() <= 0) {
      out.setZero();
      return;
    }
    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int32 num_threads = worker_threads->num_threads;
    const int64 cost_per_traverse = 500;
    auto traverse = [this, &out, &data_set, decision_tree_resource, batch_size](
                        int64 start, int64 end) {
      CHECK(start <= end);
      CHECK(end <= batch_size);
      for (int example_id = start; example_id < end; ++example_id) {
        const int32 leaf_id =
            decision_tree_resource->TraverseTree(&data_set, example_id);
        set_output_value(example_id, leaf_id, decision_tree_resource, &out);
      };
    };
    Shard(num_threads, worker_threads->workers, batch_size, cost_per_traverse,
          traverse);
  };

  void set_output_value(const int32 example_id, const int32 leaf_id,
                        const TensorForestTreeResource* decision_tree_resource,
                        TTypes<float>::Matrix* out) const {
    for (int j = 0; j < logits_dimension_; ++j) {
      const float logit = decision_tree_resource->get_prediction(leaf_id, j);
      (*out)(example_id, j) = logit;
    }
  };

 private:
  int32 logits_dimension_;
};

REGISTER_KERNEL_BUILDER(Name("TensorForestTreePredict").Device(DEVICE_CPU),
                        TensorForestTreePredictOp);

}  // namespace tensorflow
