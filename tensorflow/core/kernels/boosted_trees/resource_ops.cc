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

#include <memory>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/boosted_trees/resources.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

REGISTER_RESOURCE_HANDLE_KERNEL(BoostedTreesEnsembleResource);

REGISTER_KERNEL_BUILDER(
    Name("IsBoostedTreesEnsembleInitialized").Device(DEVICE_CPU),
    IsResourceInitialized<BoostedTreesEnsembleResource>);

// Creates a tree ensemble resource.
class BoostedTreesCreateEnsembleOp : public OpKernel {
 public:
  explicit BoostedTreesCreateEnsembleOp(OpKernelConstruction* context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
  }

  void Compute(OpKernelContext* context) override {
    // Get the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(stamp_token_t->shape()),
                errors::InvalidArgument(
                    "stamp_token must be a scalar, got a tensor of shape ",
                    stamp_token_t->shape().DebugString()));
    int64_t stamp_token = stamp_token_t->scalar<int64_t>()();

    // Get the tree ensemble proto.
    const Tensor* tree_ensemble_serialized_t;
    OP_REQUIRES_OK(context, context->input("tree_ensemble_serialized",
                                           &tree_ensemble_serialized_t));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsScalar(tree_ensemble_serialized_t->shape()),
        errors::InvalidArgument(
            "tree_ensemble_serialized must be a scalar, got a tensor of shape ",
            tree_ensemble_serialized_t->shape().DebugString()));
    std::unique_ptr<BoostedTreesEnsembleResource> result(
        new BoostedTreesEnsembleResource());
    if (!result->InitFromSerialized(
            tree_ensemble_serialized_t->scalar<tstring>()(), stamp_token)) {
      result->Unref();
      result.release();  // Needed due to the `->Unref` above, to prevent UAF
      OP_REQUIRES(
          context, false,
          errors::InvalidArgument("Unable to parse tree ensemble proto."));
    }

    // Only create one, if one does not exist already. Report status for all
    // other exceptions.
    auto status =
        CreateResource(context, HandleFromInput(context, 0), result.release());
    if (status.code() != tensorflow::error::ALREADY_EXISTS) {
      OP_REQUIRES_OK(context, status);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BoostedTreesCreateEnsemble").Device(DEVICE_CPU),
                        BoostedTreesCreateEnsembleOp);

// Op for retrieving some model states (needed for training).
class BoostedTreesGetEnsembleStatesOp : public OpKernel {
 public:
  explicit BoostedTreesGetEnsembleStatesOp(OpKernelConstruction* context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
  }

  void Compute(OpKernelContext* context) override {
    // Looks up the resource.
    core::RefCountPtr<BoostedTreesEnsembleResource> tree_ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &tree_ensemble_resource));
    tf_shared_lock l(*tree_ensemble_resource->get_mutex());

    // Sets the outputs.
    const int num_trees = tree_ensemble_resource->num_trees();
    const int num_finalized_trees =
        (num_trees <= 0 ||
         tree_ensemble_resource->IsTreeFinalized(num_trees - 1))
            ? num_trees
            : num_trees - 1;
    const int num_attempted_layers =
        tree_ensemble_resource->GetNumLayersAttempted();

    // growing_metadata
    Tensor* output_stamp_token_t = nullptr;
    Tensor* output_num_trees_t = nullptr;
    Tensor* output_num_finalized_trees_t = nullptr;
    Tensor* output_num_attempted_layers_t = nullptr;
    Tensor* output_last_layer_nodes_range_t = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
                                                     &output_stamp_token_t));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape(),
                                                     &output_num_trees_t));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape(),
                                            &output_num_finalized_trees_t));
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, TensorShape(),
                                            &output_num_attempted_layers_t));
    OP_REQUIRES_OK(context, context->allocate_output(
                                4, {2}, &output_last_layer_nodes_range_t));

    output_stamp_token_t->scalar<int64_t>()() = tree_ensemble_resource->stamp();
    output_num_trees_t->scalar<int32>()() = num_trees;
    output_num_finalized_trees_t->scalar<int32>()() = num_finalized_trees;
    output_num_attempted_layers_t->scalar<int32>()() = num_attempted_layers;

    int32_t range_start;
    int32_t range_end;
    tree_ensemble_resource->GetLastLayerNodesRange(&range_start, &range_end);

    output_last_layer_nodes_range_t->vec<int32>()(0) = range_start;
    // For a completely empty ensemble, this will be 0. To make it a valid range
    // we add this max cond.
    output_last_layer_nodes_range_t->vec<int32>()(1) = std::max(1, range_end);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesGetEnsembleStates").Device(DEVICE_CPU),
    BoostedTreesGetEnsembleStatesOp);

// Op for serializing a model.
class BoostedTreesSerializeEnsembleOp : public OpKernel {
 public:
  explicit BoostedTreesSerializeEnsembleOp(OpKernelConstruction* context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
  }

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<BoostedTreesEnsembleResource> tree_ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &tree_ensemble_resource));
    tf_shared_lock l(*tree_ensemble_resource->get_mutex());
    Tensor* output_stamp_token_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
                                                     &output_stamp_token_t));
    output_stamp_token_t->scalar<int64_t>()() = tree_ensemble_resource->stamp();
    Tensor* output_proto_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape(), &output_proto_t));
    output_proto_t->scalar<tstring>()() =
        tree_ensemble_resource->SerializeAsString();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesSerializeEnsemble").Device(DEVICE_CPU),
    BoostedTreesSerializeEnsembleOp);

// Op for deserializing a tree ensemble variable from a checkpoint.
class BoostedTreesDeserializeEnsembleOp : public OpKernel {
 public:
  explicit BoostedTreesDeserializeEnsembleOp(OpKernelConstruction* context)
      : OpKernel(context) {
    VLOG(1) << "Boosted Trees kernels in TF are deprecated. Please use "
            << "TensorFlow Decision Forests instead "
            << "(https://github.com/tensorflow/decision-forests).\n";
  }

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<BoostedTreesEnsembleResource> tree_ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &tree_ensemble_resource));
    mutex_lock l(*tree_ensemble_resource->get_mutex());

    // Get the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(stamp_token_t->shape()),
                errors::InvalidArgument(
                    "stamp_token must be a scalar, got a tensor of shape ",
                    stamp_token_t->shape().DebugString()));
    int64_t stamp_token = stamp_token_t->scalar<int64_t>()();

    // Get the tree ensemble proto.
    const Tensor* tree_ensemble_serialized_t;
    OP_REQUIRES_OK(context, context->input("tree_ensemble_serialized",
                                           &tree_ensemble_serialized_t));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsScalar(tree_ensemble_serialized_t->shape()),
        errors::InvalidArgument(
            "tree_ensemble_serialized must be a scalar, got a tensor of shape ",
            tree_ensemble_serialized_t->shape().DebugString()));
    // Deallocate all the previous objects on the resource.
    tree_ensemble_resource->Reset();
    OP_REQUIRES(
        context,
        tree_ensemble_resource->InitFromSerialized(
            tree_ensemble_serialized_t->scalar<tstring>()(), stamp_token),
        errors::InvalidArgument("Unable to parse tree ensemble proto."));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BoostedTreesDeserializeEnsemble").Device(DEVICE_CPU),
    BoostedTreesDeserializeEnsembleOp);

}  // namespace tensorflow
