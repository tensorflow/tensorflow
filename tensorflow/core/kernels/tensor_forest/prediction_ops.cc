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
#include "tensorflow/core/kernels/tensor_forest/leaf_model.h"
#include "tensorflow/core/kernels/tensor_forest/resources.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

void TraverseTree(const DecisionTreeResource* tree_resource,
                  const std::unique_ptr<DenseTensorType>& data, int32 start,
                  int32 end,
                  const std::function<void(int32, int32)>& set_leaf_id,
                  std::vector<tensorforest::TreePath>* tree_paths) {
  for (int i = start; i < end; ++i) {
    const int32 id = tree_resource->TraverseTree(
        data, i, (tree_paths == nullptr) ? nullptr : &(*tree_paths)[i]);
    set_leaf_id(i, id);
  }
};

class TreePredictionsOp : public OpKernel {
 public:
  explicit TreePredictionsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("leaf_model_type", &leaf_model_type_));

    OP_REQUIRES_OK(context, context->GetAttr("num_output", &num_output_));
    model_op_ = LeafModelFactory::CreateLeafModelOperator(leaf_model_type_,
                                                          num_output_);
  }

  void Compute(OpKernelContext* context) override {
    DecisionTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const Tensor& input_data = context->input(1);

    std::unique_ptr<DenseTensorType> data_set = nullptr;
    data_set.reset(new DenseTensorType(input_data.tensor<float, 2>()));
    const int num_data = input_data.dim_size(0);

    Tensor* output_predictions = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(num_data);
    output_shape.AddDim(num_output_);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_predictions));
    TTypes<float, 2>::Tensor out = output_predictions->tensor<float, 2>();

    std::vector<tensorforest::TreePath> tree_paths(num_data);

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;
    const int64 costPerTraverse = 500;
    auto traverse = [this, &out, &data_set, decision_tree_resource, num_data,
                     &tree_paths](int64 start, int64 end) {
      CHECK(start <= end);
      CHECK(end <= num_data);
      TraverseTree(decision_tree_resource, data_set, static_cast<int32>(start),
                   static_cast<int32>(end),
                   std::bind(&TreePredictionsOp::set_output_value, this,
                             std::placeholders::_1, std::placeholders::_2,
                             decision_tree_resource, &out),
                   &tree_paths);
    };
    Shard(num_threads, worker_threads->workers, num_data, costPerTraverse,
          traverse);

    Tensor* output_tree_paths = nullptr;
    TensorShape output_paths_shape;
    output_paths_shape.AddDim(num_data);
    OP_REQUIRES_OK(context, context->allocate_output(1, output_paths_shape,
                                                     &output_tree_paths));
    auto out_paths = output_tree_paths->unaligned_flat<string>();

    // TODO(gilberth): If this slows down inference too much, consider having
    // a filter that only serializes paths for the predicted label that we're
    // interested in.
    for (int i = 0; i < tree_paths.size(); ++i) {
      out_paths(i) = tree_paths[i].SerializeAsString();
    }
  }

  void set_output_value(int32 i, int32 id,
                        DecisionTreeResource* decision_tree_resource,
                        TTypes<float, 2>::Tensor* out) {
    const tensorforest::Leaf& leaf = decision_tree_resource->get_leaf(id);

    float sum = 0;
    for (int j = 0; j < num_output_; ++j) {
      const float count = model_op_->GetOutputValue(leaf, j);
      (*out)(i, j) = count;
      sum += count;
    }

    if (leaf_model_type_ != static_cast<int32>(LeafModelType::CLASSIFICATION) &&
        sum > 0 && sum != 1) {
      for (int j = 0; j < num_output_; ++j) {
        (*out)(i, j) /= sum;
      }
    }
  }

 private:
  std::unique_ptr<LeafModelOperator> model_op_;
  int32 leaf_model_type_;
  int32 num_output_;
};

REGISTER_KERNEL_BUILDER(Name("TreePredictions").Device(DEVICE_CPU),
                        TreePredictionsOp);

}  // namespace tensorflow
