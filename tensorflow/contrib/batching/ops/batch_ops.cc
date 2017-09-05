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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("Batch")
    .Input("in_tensors: T")
    .Output("batched_tensors: T")
    .Output("batch_index: int64")
    .Output("id: int64")
    .Attr("num_batch_threads: int")
    .Attr("max_batch_size: int")
    .Attr("batch_timeout_micros: int")
    .Attr("allowed_batch_sizes: list(int) = []")
    .Attr("grad_timeout_micros: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("batching_queue: string = ''")
    .Attr("T: list(type)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<shape_inference::ShapeHandle> in_shapes;
      TF_RETURN_IF_ERROR(c->input("in_tensors", &in_shapes));
      std::vector<shape_inference::ShapeHandle> out_shapes(in_shapes.size());
      for (int i = 0; i < in_shapes.size(); ++i) {
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(in_shapes[i], 0, c->UnknownDim(), &out_shapes[i]));
      }
      TF_RETURN_IF_ERROR(c->set_output("batched_tensors", out_shapes));
      TF_RETURN_IF_ERROR(c->set_output("id", {c->Scalar()}));
      TF_RETURN_IF_ERROR(c->set_output(
          "batch_index",
          {c->MakeShape({shape_inference::DimensionOrConstant(c->UnknownDim()),
                         shape_inference::DimensionOrConstant(3)})}));
      return Status::OK();
    })
    .Doc(R"doc(
Batches all input tensors nondeterministically.

When many instances of this Op are being run concurrently with the same
container/shared_name in the same device, some will output zero-shaped Tensors
and others will output Tensors of size up to max_batch_size.

All Tensors in in_tensors are batched together (so, for example, labels and
features should be batched with a single instance of this operation.

Each invocation of batch emits an `id` scalar which will be used to identify
this particular invocation when doing unbatch or its gradient.

Each op which emits a non-empty batch will also emit a non-empty batch_index
Tensor, which, is a [K, 3] matrix where each row contains the invocation's id,
start, and length of elements of each set of Tensors present in batched_tensors.

Batched tensors are concatenated along the first dimension, and all tensors in
in_tensors must have the first dimension of the same size.

in_tensors: The tensors to be batched.
num_batch_threads: Number of scheduling threads for processing batches of work.
 Determines the number of batches processed in parallel.
max_batch_size: Batch sizes will never be bigger than this.
batch_timeout_micros: Maximum number of microseconds to wait before outputting
 an incomplete batch.
allowed_batch_sizes: Optional list of allowed batch sizes. If left empty, does
 nothing. Otherwise, supplies a list of batch sizes, causing the op to pad
 batches up to one of those sizes. The entries must increase monotonically, and
 the final entry must equal max_batch_size.
grad_timeout_micros: The timeout to use for the gradient. See Unbatch.
batched_tensors: Either empty tensors or a batch of concatenated Tensors.
batch_index: If out_tensors is non-empty, has information to invert it.
container: Controls the scope of sharing of this batch.
id: always contains a scalar with a unique ID for this invocation of Batch.
shared_name: Concurrently running instances of batch in the same device with the
 same container and shared_name will batch their elements together. If left
 empty, the op name will be used as the shared name.
T: the types of tensors to be batched.
)doc");

REGISTER_OP("Unbatch")
    .Input("batched_tensor: T")
    .Input("batch_index: int64")
    .Input("id: int64")
    .Output("unbatched_tensor: T")
    .Attr("timeout_micros: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle out_shape;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &out_shape));
      c->set_output(0, out_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Reverses the operation of Batch for a single output Tensor.

An instance of Unbatch either receives an empty batched_tensor, in which case it
asynchronously waits until the values become available from a concurrently
running instance of Unbatch with the same container and shared_name, or receives
a non-empty batched_tensor in which case it finalizes all other concurrently
running instances and outputs its own element from the batch.

batched_tensor: The possibly transformed output of Batch. The size of the first
 dimension should remain unchanged by the transformations for the operation to
 work.
batch_index: The matching batch_index obtained from Batch.
id: The id scalar emitted by Batch.
unbatched_tensor: The Tensor corresponding to this execution.
timeout_micros: Maximum amount of time (in microseconds) to wait to receive the
 batched input tensor associated with a given invocation of the op.
container: Container to control resource sharing.
shared_name: Instances of Unbatch with the same container and shared_name are
 assumed to possibly belong to the same batch. If left empty, the op name will
 be used as the shared name.
)doc");

REGISTER_OP("UnbatchGrad")
    .Input("original_input: T")
    .Input("batch_index: int64")
    .Input("grad: T")
    .Input("id: int64")
    .Output("batched_grad: T")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(c->Rank(c->input(2))));
      return Status::OK();
    })
    .Doc(R"doc(
Gradient of Unbatch.

Acts like Batch but using the given batch_index index of batching things as they
become available. This ensures that the gradients are propagated back in the
same session which did the forward pass.

original_input: The input to the Unbatch operation this is the gradient of.
batch_index: The batch_index given to the Unbatch operation this is the gradient
of.
grad: The downstream gradient.
id: The id scalar emitted by Batch.
batched_grad: The return value, either an empty tensor or the batched gradient.
container: Container to control resource sharing.
shared_name: Instances of UnbatchGrad with the same container and shared_name
 are assumed to possibly belong to the same batch. If left empty, the op name
 will be used as the shared name.
  )doc");

}  // namespace tensorflow
