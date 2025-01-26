/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/data_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"

namespace tensorflow {
namespace ops {
namespace {

REGISTER_NO_GRADIENT_OP("Queue");
REGISTER_NO_GRADIENT_OP("QueueEnqueue");
REGISTER_NO_GRADIENT_OP("QueueEnqueueMany");
REGISTER_NO_GRADIENT_OP("QueueDequeue");
REGISTER_NO_GRADIENT_OP("QueueDequeueMany");
REGISTER_NO_GRADIENT_OP("QueueDequeueUpTo");
REGISTER_NO_GRADIENT_OP("QueueClose");
REGISTER_NO_GRADIENT_OP("QueueSize");
REGISTER_NO_GRADIENT_OP("Stack");
REGISTER_NO_GRADIENT_OP("StackPush");
REGISTER_NO_GRADIENT_OP("StackPop");
REGISTER_NO_GRADIENT_OP("StackClose");
REGISTER_NO_GRADIENT_OP("GetSessionHandle");
REGISTER_NO_GRADIENT_OP("GetSessionHandleV2");
REGISTER_NO_GRADIENT_OP("GetSessionTensor");
REGISTER_NO_GRADIENT_OP("DeleteSessionTensor");

absl::Status DynamicPartitionGrad(const Scope& scope, const Operation& op,
                                  const std::vector<Output>& grad_inputs,
                                  std::vector<Output>* grad_outputs) {
  // DynamicPartition only moves input values into various positions
  // in the output, so the gradient operation only has to map incoming
  // gradients into their input source locations.
  // running example:
  // data = [10, 20, 30, 40, 50]
  // partitions = [0, 0, 1, 1, 0]
  // num_partitions = 2
  // dynamic_partition(data, partitions, num_partitions) = {
  //   [10, 20, 50],
  //   [30, 40]
  // }
  // grads = {
  //   [g1, g2, g3],
  //   [g4, g5]
  // }
  // The desired propagation of the gradients back to the data inputs is:
  // [g1, g2, g4, g5, g3]
  auto data = op.input(0);
  auto partitions = op.input(1);
  int32_t num_partitions;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "num_partitions", &num_partitions));

  // Note: the shape of the partitions is a prefix of the data shape.
  // shape(partitions) = [5]
  auto partitions_shape = Shape(scope, partitions);
  // We now create a partitions-shaped tensor with integers from
  // [0..size(partitions)) This will be dynamic_partitioned with the
  // input parameters, providing the destination index for a given
  // source item.
  // partitions_size = prod([5]) = 5
  // reshape(range(partitions_size), [5]) = [0, 1, 2, 3, 4]
  auto zero = Const(scope, 0);
  auto one = Const(scope, 1);
  auto original_indices = Reshape(
      scope, Range(scope, zero, Prod(scope, partitions_shape, zero), one),
      partitions_shape);
  // dynamic_partition(
  //   [0, 1, 2, 3, 4],
  //   [0, 0, 1, 1, 0], 2)
  //  = { [0, 1, 4],
  //      [2, 3] }
  auto partitioned_indices =
      DynamicPartition(scope, original_indices, partitions, num_partitions);

  // Invert these indices with dynamic_stitch to map the incoming
  // gradients to their source inputs.
  // dynamic_stitch(
  //   { [0, 1, 4], [2, 3] },
  //   { [g1, g2, g3], [g4, g5] })
  // = [g1, g2, g4, g5, g3]
  auto reconstructed =
      DynamicStitch(scope, partitioned_indices.outputs, grad_inputs);
  // reshape back into a data-shaped tensor to propagate gradients for the data
  // input.
  grad_outputs->push_back(Reshape(scope, reconstructed, Shape(scope, data)));
  // Stop propagation along the partitions input
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("DynamicPartition", DynamicPartitionGrad);

absl::Status DynamicStitchGrad(const Scope& scope, const Operation& op,
                               const std::vector<Output>& grad_inputs,
                               std::vector<Output>* grad_outputs) {
  // Running example:
  // indices = {2, [1, 0]}
  // data = {[d_1, d_2], [[d_3, d_4], [d_5, d_6]]}
  // out = [[d_5, d_6], [d_3, d_4], [d_1, d_2]]
  // grad = [[g_1, g_2], [g_3, g_4], [g_5, g_6]]

  // indices and data are two equal-sized lists passed
  // into DynamicStitch.
  // num_values = 2
  int32_t num_values = op.num_inputs() / 2;

  // Stop propagation along the indices list
  for (int32_t i = 0; i < num_values; i++) {
    grad_outputs->push_back(NoGradient());
  }

  // DynamicStitch shuffles its data to the output (using items in
  // indices) so the gradient propagated to a given data input simply
  // selects the gradient for its output position.
  for (int32_t i = 0; i < num_values; i++) {
    // index has the destination positions for the i'th data
    // element. We cast it into an int32 if necessary, so we can use
    // it from a Gather op.
    // i = 0: index = 2
    // i = 1: index = [1, 0]
    auto index = op.input(i);
    if (index.type() != DT_INT32) {
      index = Cast(scope, index, DT_INT32);
    }
    // Gather the index specified locations in the gradient and
    // propagate it as the gradient for the i'th data item.
    // i = 0: gather(grad, 2) = [g_5, g_6]
    // i = 1: gather(grad, [1, 0]) = [[g_3, g_4], [g_1, g_2]]
    grad_outputs->push_back(Gather(scope, grad_inputs[0], index));
  }

  return scope.status();
}
REGISTER_GRADIENT_OP("DynamicStitch", DynamicStitchGrad);
REGISTER_GRADIENT_OP("ParallelDynamicStitch", DynamicStitchGrad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
