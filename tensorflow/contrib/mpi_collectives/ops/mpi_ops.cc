/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_MPI

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace contrib {
namespace mpi_collectives {

REGISTER_OP("MPIInit").Doc(R"doc(
Initialize MPI for the current process.

If this is run on a GPU, then that GPU must be used for all future MPI
operations. If it is run on CPU, then all future MPI operations must also
run on CPU.
)doc");

REGISTER_OP("MPISize")
    .Output("size: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the number of running MPI processes.

More precisely, returns the number of MPI processes in the group associated
with the MPI_COMM_WORLD communicator.

size:   Size of the MPI group.
)doc");

REGISTER_OP("MPIRank")
    .Output("rank: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the index of the current process in the MPI group.

More precisely, returns the rank of the calling process in the MPI_COMM_WORLD
communicator.

rank:   Rank of the calling process.
)doc");

REGISTER_OP("MPILocalRank")
    .Output("rank: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the index of the current process in the node it is on.

More precisely, returns the rank of the calling process in communicator that
only spans the MPI processes running on that node.

rank:   Rank of the calling process on the node it is on.
)doc");

REGISTER_OP("MPIAllreduce")
    .Attr("T: {int32, int64, float32}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
allreduce.

Arguments
    tensor:     A tensor to reduce.

Output
    sum:        A tensor with the same shape as `tensor`, summed across all
                MPI processes.
)doc");

REGISTER_OP("MPIAllgather")
    .Attr("T: {int32, int64, float32}")
    .Attr("S: {int64}")
    .Input("tensor: T")
    .Input("sizes: S")
    .Output("gathered: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allgather on a tensor. All other processes that do a gather on a
tensor with the same name must have the same rank for that tensor, and have the
same dimension on all but the first dimension.

Arguments
    tensor:     A tensor to gather.
    sizes:      A tensor containing the first-dimension sizes of tensors to be
                gathered from other ranks

Output
    gathered:   A tensor with the same shape as `tensor` except for the first
                dimension, which is the sum of dimensions in `sizes`.
)doc");

}  // namespace mpi_collectives
}  // namespace contrib
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_MPI
