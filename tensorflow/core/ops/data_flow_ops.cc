/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// --------------------------------------------------------------------------

REGISTER_OP("DynamicPartition")
    .Input("data: T")
    .Input("partitions: int32")
    .Output("outputs: num_partitions * T")
    .Attr("num_partitions: int")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      int64 num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));

      ShapeHandle data_shape = c->input(0);
      ShapeHandle partitions_shape = c->input(1);

      if (!c->RankKnown(partitions_shape)) {
        return shape_inference::UnknownShape(c);
      }

      const int64 rank = c->Rank(partitions_shape);

      // data shape must start with partitions_shape
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->MergePrefix(data_shape, partitions_shape, &unused, &unused));

      // The partition shape is dynamic in the 0th dimension, and matches
      // data_shape in the remaining dimensions.
      ShapeHandle unknown_dim0 = c->MakeShape({c->UnknownDim()});

      ShapeHandle data_suffix_shape;
      TF_RETURN_IF_ERROR(c->Subshape(data_shape, rank, &data_suffix_shape));
      ShapeHandle result_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(unknown_dim0, data_suffix_shape, &result_shape));

      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, result_shape);
      }

      return Status::OK();
    })
    .Doc(R"doc(
Partitions `data` into `num_partitions` tensors using indices from `partitions`.

For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
are placed in `outputs[i]` in lexicographic order of `js`, and the first
dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
In detail,

```python
    outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

    outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
```

`data.shape` must start with `partitions.shape`.

For example:

```python
    # Scalar partitions.
    partitions = 1
    num_partitions = 2
    data = [10, 20]
    outputs[0] = []  # Empty with shape [0, 2]
    outputs[1] = [[10, 20]]

    # Vector partitions.
    partitions = [0, 0, 1, 1, 0]
    num_partitions = 2
    data = [10, 20, 30, 40, 50]
    outputs[0] = [10, 20, 50]
    outputs[1] = [30, 40]
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/DynamicPartition.png" alt>
</div>

partitions: Any shape.  Indices in the range `[0, num_partitions)`.
num_partitions: The number of partitions to output.
)doc");

REGISTER_OP("DynamicStitch")
    .Input("indices: N * int32")
    .Input("data: N * T")
    .Output("merged: T")
    .Attr("N : int >= 1")
    .Attr("T : type")
    .SetShapeFn([](InferenceContext* c) {
      int64 num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_partitions));

      ShapeHandle extra_shape = c->UnknownShape();
      for (int i = 0; i < num_partitions; ++i) {
        ShapeHandle indices_shape = c->input(i);
        ShapeHandle data_shape = c->input(i + num_partitions);
        if (!c->RankKnown(indices_shape)) {
          continue;
        }

        const int64 indices_rank = c->Rank(indices_shape);

        // Assert that data_shape starts with indices_shape.
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(
            c->MergePrefix(data_shape, indices_shape, &unused, &unused));

        // The rest belongs to output.
        ShapeHandle rest;
        TF_RETURN_IF_ERROR(c->Subshape(data_shape, indices_rank, &rest));
        TF_RETURN_IF_ERROR(c->Merge(extra_shape, rest, &extra_shape));
      }

      ShapeHandle output_shape = c->Vector(c->UnknownDim());
      TF_RETURN_IF_ERROR(
          c->Concatenate(output_shape, extra_shape, &output_shape));
      c->set_output(0, output_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Interleave the values from the `data` tensors into a single tensor.

Builds a merged tensor such that

```python
    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
```

For example, if each `indices[m]` is scalar or vector, we have

```python
    # Scalar indices:
    merged[indices[m], ...] = data[m][...]

    # Vector indices:
    merged[indices[m][i], ...] = data[m][i, ...]
```

Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is

    merged.shape = [max(indices)] + constant

Values are merged in order, so if an index appears in both `indices[m][i]` and
`indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
merged result.

For example:

```python
    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/DynamicStitch.png" alt>
</div>
)doc");

// --------------------------------------------------------------------------

namespace {
Status TwoElementVectorInputsAndScalarOutputs(InferenceContext* c) {
  ShapeHandle handle;
  DimensionHandle unused_handle;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status TwoElementOutput(InferenceContext* c) {
  c->set_output(0, c->Vector(2));
  return Status::OK();
}
}  // namespace

REGISTER_OP("RandomShuffleQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("min_after_dequeue: int = 0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
A queue that randomizes the order of elements.

handle: The handle to the queue.
component_types: The type of each component in a value.
shapes: The shape of each component in a value. The length of this attr must
  be either 0 or the same as the length of component_types. If the length of
  this attr is 0, the shapes of queue elements are not constrained, and
  only one element may be dequeued at a time.
capacity: The upper bound on the number of elements in this queue.
  Negative numbers mean no limit.
min_after_dequeue: Dequeue will block unless there would be this
  many elements after the dequeue or the queue is closed. This
  ensures a minimum level of mixing of elements.
seed: If either seed or seed2 is set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, a random seed is used.
seed2: A second seed to avoid seed collision.
container: If non-empty, this queue is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this queue will be shared under the given name
  across multiple sessions.
)doc");

REGISTER_OP("FIFOQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
A queue that produces elements in first-in first-out order.

handle: The handle to the queue.
component_types: The type of each component in a value.
shapes: The shape of each component in a value. The length of this attr must
  be either 0 or the same as the length of component_types. If the length of
  this attr is 0, the shapes of queue elements are not constrained, and
  only one element may be dequeued at a time.
capacity: The upper bound on the number of elements in this queue.
  Negative numbers mean no limit.
container: If non-empty, this queue is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this queue will be shared under the given name
  across multiple sessions.
)doc");

REGISTER_OP("PaddingFIFOQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
A queue that produces elements in first-in first-out order.

Variable-size shapes are allowed by setting the corresponding shape dimensions
to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
size of any given element in the minibatch.  See below for details.

handle: The handle to the queue.
component_types: The type of each component in a value.
shapes: The shape of each component in a value. The length of this attr must
  be either 0 or the same as the length of component_types.
  Shapes of fixed rank but variable size are allowed by setting
  any shape dimension to -1.  In this case, the inputs' shape may vary along
  the given dimension, and DequeueMany will pad the given dimension with
  zeros up to the maximum shape of all elements in the given batch.
  If the length of this attr is 0, different queue elements may have
  different ranks and shapes, but only one element may be dequeued at a time.
capacity: The upper bound on the number of elements in this queue.
  Negative numbers mean no limit.
container: If non-empty, this queue is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this queue will be shared under the given name
  across multiple sessions.
)doc");

REGISTER_OP("PriorityQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 0 = []")
    .Attr("shapes: list(shape) >= 0")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
A queue that produces elements sorted by the first component value.

Note that the PriorityQueue requires the first component of any element
to be a scalar int64, in addition to the other elements declared by
component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
entry in their input (resp. output) lists.

handle: The handle to the queue.
component_types: The type of each component in a value.
shapes: The shape of each component in a value. The length of this attr must
  be either 0 or the same as the length of component_types. If the length of
  this attr is 0, the shapes of queue elements are not constrained, and
  only one element may be dequeued at a time.
capacity: The upper bound on the number of elements in this queue.
  Negative numbers mean no limit.
container: If non-empty, this queue is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this queue will be shared under the given name
  across multiple sessions.
)doc");

REGISTER_OP("QueueEnqueue")
    .Input("handle: Ref(string)")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Enqueues a tuple of one or more tensors in the given queue.

The components input has k elements, which correspond to the components of
tuples stored in the given queue.

N.B. If the queue is full, this operation will block until the given
element has been enqueued (or 'timeout_ms' elapses, if specified).

handle: The handle to a queue.
components: One or more tensors from which the enqueued tensors should be taken.
timeout_ms: If the queue is full, this operation will block for up to
  timeout_ms milliseconds.
  Note: This option is not supported yet.
)doc");

REGISTER_OP("QueueEnqueueMany")
    .Input("handle: Ref(string)")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Enqueues zero or more tuples of one or more tensors in the given queue.

This operation slices each component tensor along the 0th dimension to
make multiple queue elements. All of the tuple components must have the
same size in the 0th dimension.

The components input has k elements, which correspond to the components of
tuples stored in the given queue.

N.B. If the queue is full, this operation will block until the given
elements have been enqueued (or 'timeout_ms' elapses, if specified).

handle: The handle to a queue.
components: One or more tensors from which the enqueued tensors should
  be taken.
timeout_ms: If the queue is too full, this operation will block for up
  to timeout_ms milliseconds.
  Note: This option is not supported yet.
)doc");

REGISTER_OP("QueueDequeue")
    .Input("handle: Ref(string)")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Dequeues a tuple of one or more tensors from the given queue.

This operation has k outputs, where k is the number of components
in the tuples stored in the given queue, and output i is the ith
component of the dequeued tuple.

N.B. If the queue is empty, this operation will block until an element
has been dequeued (or 'timeout_ms' elapses, if specified).

handle: The handle to a queue.
components: One or more tensors that were dequeued as a tuple.
component_types: The type of each component in a tuple.
timeout_ms: If the queue is empty, this operation will block for up to
  timeout_ms milliseconds.
  Note: This option is not supported yet.
)doc");

REGISTER_OP("QueueDequeueMany")
    .Input("handle: Ref(string)")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Dequeues n tuples of one or more tensors from the given queue.

If the queue is closed and there are fewer than n elements, then an
OutOfRange error is returned.

This operation concatenates queue-element component tensors along the
0th dimension to make a single component tensor.  All of the components
in the dequeued tuple will have size n in the 0th dimension.

This operation has k outputs, where k is the number of components in
the tuples stored in the given queue, and output i is the ith
component of the dequeued tuple.

N.B. If the queue is empty, this operation will block until n elements
have been dequeued (or 'timeout_ms' elapses, if specified).

handle: The handle to a queue.
n: The number of tuples to dequeue.
components: One or more tensors that were dequeued as a tuple.
component_types: The type of each component in a tuple.
timeout_ms: If the queue has fewer than n elements, this operation
  will block for up to timeout_ms milliseconds.
  Note: This option is not supported yet.
)doc");

REGISTER_OP("QueueDequeueUpTo")
    .Input("handle: Ref(string)")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Dequeues n tuples of one or more tensors from the given queue.

This operation is not supported by all queues.  If a queue does not support
DequeueUpTo, then an Unimplemented error is returned.

If the queue is closed and there are more than 0 but less than n elements
remaining, then instead of returning an OutOfRange error like
QueueDequeueMany, less than `n` elements are returned immediately.  If the queue
is closed and there are 0 elements left in the queue, then an OutOfRange
error is returned just like in QueueDequeueMany.  Otherwise the behavior
is identical to QueueDequeueMany:

This operation concatenates queue-element component tensors along the
0th dimension to make a single component tensor.  All of the components
in the dequeued tuple will have size n in the 0th dimension.

This operation has k outputs, where k is the number of components in
the tuples stored in the given queue, and output i is the ith
component of the dequeued tuple.

handle: The handle to a queue.
n: The number of tuples to dequeue.
components: One or more tensors that were dequeued as a tuple.
component_types: The type of each component in a tuple.
timeout_ms: If the queue has fewer than n elements, this operation
  will block for up to timeout_ms milliseconds.
  Note: This option is not supported yet.
)doc");

REGISTER_OP("QueueClose")
    .Input("handle: Ref(string)")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Attr("cancel_pending_enqueues: bool = false")
    .Doc(R"doc(
Closes the given queue.

This operation signals that no more elements will be enqueued in the
given queue. Subsequent Enqueue(Many) operations will fail.
Subsequent Dequeue(Many) operations will continue to succeed if
sufficient elements remain in the queue. Subsequent Dequeue(Many)
operations that would block will fail immediately.

handle: The handle to a queue.
cancel_pending_enqueues: If true, all pending enqueue requests that are
  blocked on the given queue will be cancelled.
)doc");

REGISTER_OP("QueueSize")
    .Input("handle: Ref(string)")
    .Output("size: int32")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Doc(R"doc(
Computes the number of elements in the given queue.

handle: The handle to a queue.
size: The number of elements in the given queue.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("AccumulatorNumAccumulated")
    .Input("handle: Ref(string)")
    .Output("num_accumulated: int32")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Returns the number of gradients aggregated in the given accumulators.

handle: The handle to an accumulator.
num_accumulated: The number of gradients aggregated in the given accumulator.
)doc");

REGISTER_OP("AccumulatorSetGlobalStep")
    .Input("handle: Ref(string)")
    .Input("new_global_step: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    })
    .Doc(R"doc(
Updates the accumulator with a new value for global_step. Logs warning if the
accumulator's value is already higher than new_global_step.

handle: The handle to an accumulator.
new_global_step: The new global_step value to set.
)doc");

REGISTER_OP("ConditionalAccumulator")
    .Output("handle: Ref(string)")
    .Attr("dtype: numbertype")
    .Attr("shape: shape")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
    .Doc(R"doc(
A conditional accumulator for aggregating gradients. The accumulator accepts
gradients marked with local_step greater or equal to the most recent global_step
known to the accumulator. The average can be extracted from the accumulator,
provided sufficient gradients have been accumulated. Extracting the average
automatically resets the aggregate to 0, and increments the global_step recorded
by the accumulator.

handle: The handle to the accumulator.
dtype: The type of the value being accumulated.
shape: The shape of the values, can be [], in which case shape is unknown.
container: If non-empty, this accumulator is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this accumulator will be shared under the given name
  across multiple sessions.
)doc");

REGISTER_OP("AccumulatorApplyGradient")
    .Input("handle: Ref(string)")
    .Input("local_step: int64")
    .Input("gradient: dtype")
    .Attr("dtype: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    })
    .Doc(R"doc(
Applies a gradient to a given accumulator. Does not add if local_step is lesser
than the accumulator's global_step.

handle: The handle to a accumulator.
local_step: The local_step value at which the gradient was computed.
gradient: A tensor of the gradient to be accumulated.
dtype: The data type of accumulated gradients. Needs to correspond to the type
  of the accumulator.
)doc");

REGISTER_OP("AccumulatorTakeGradient")
    .Input("handle: Ref(string)")
    .Input("num_required: int32")
    .Output("average: dtype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // Shape of output is the shape of the accumulator referenced
      // by 'handle', but which is not available here, so we lose
      // shape information.
      return shape_inference::UnknownShape(c);
    })
    .Attr("dtype: numbertype")
    .Doc(R"doc(
Extracts the average gradient in the given ConditionalAccumulator, provided
that sufficient (i.e., more than num_required) gradients have been accumulated.
The op blocks until sufficient gradients have been accumulated.
If the accumulator has already aggregated more than num_required gradients, it
returns the average of the accumulated gradients.
Also automatically increments the recorded global_step in the accumulator by 1,
and resets the aggregate to 0.

handle: The handle to an accumulator.
num_required: Number of gradients required before we return an aggregate.
average: The average of the accumulated gradients.
dtype: The data type of accumulated gradients. Needs to correspond to the type
  of the accumulator.
)doc");

REGISTER_OP("SparseConditionalAccumulator")
    .Output("handle: Ref(string)")
    .Attr("dtype: numbertype")
    .Attr("shape: shape")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
    .Doc(R"doc(
A conditional accumulator for aggregating sparse gradients. The accumulator
accepts gradients marked with local_step greater or equal to the most recent
global_step known to the accumulator. The average can be extracted from the
accumulator, provided sufficient gradients have been accumulated. Extracting the
average automatically resets the aggregate to 0, and increments the global_step
recorded by the accumulator.

handle: The handle to the accumulator.
dtype: The type of the value being accumulated.
shape: The shape of the values.
container: If non-empty, this accumulator is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this accumulator will be shared under the given name
  across multiple sessions.
)doc");

REGISTER_OP("SparseAccumulatorApplyGradient")
    .Input("handle: Ref(string)")
    .Input("local_step: int64")
    .Input("gradient_indices: int64")
    .Input("gradient_values: dtype")
    .Input("gradient_shape: int64")
    .Attr("dtype: numbertype")
    .Attr("has_known_shape: bool")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    })
    .Doc(R"doc(
Applies a sparse gradient to a given accumulator. Does not add if local_step is
lesser than the accumulator's global_step.

handle: The handle to a accumulator.
local_step: The local_step value at which the sparse gradient was computed.
gradient_indices: Indices of the sparse gradient to be accumulated. Must be a
  vector.
gradient_values: Values are the non-zero slices of the gradient, and must have
  the same first dimension as indices, i.e., the nnz represented by indices and
  values must be consistent.
gradient_shape: Shape of the sparse gradient to be accumulated.
dtype: The data type of accumulated gradients. Needs to correspond to the type
  of the accumulator.
has_known_shape: Boolean indicating whether gradient_shape is unknown, in which
  case the input is ignored during validation.
)doc");

REGISTER_OP("SparseAccumulatorTakeGradient")
    .Input("handle: Ref(string)")
    .Input("num_required: int32")
    .Output("indices: int64")
    .Output("values: dtype")
    .Output("shape: int64")
    .Attr("dtype: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // Shape of output is the shape of the accumulator referenced
      // by 'handle', but which is not available here, so we lose
      // shape information.
      return shape_inference::UnknownShape(c);
    })
    .Doc(R"doc(
Extracts the average sparse gradient in the given SparseConditionalAccumulator,
provided that sufficient (i.e., more than num_required) gradients have been
accumulated. The op will blocks until sufficient gradients have been
accumulated. If the accumulator has already aggregated more than num_required
gradients, it will return its average of the accumulated gradients.
Also automatically increments the recorded global_step in the accumulator by 1,
and resets the aggregate to 0.

handle: The handle to a SparseConditionalAccumulator.
num_required: Number of gradients required before we return an aggregate.
indices: Indices of the average of the accumulated sparse gradients.
values: Values of the average of the accumulated sparse gradients.
shape: Shape of the average of the accumulated sparse gradients.
dtype: The data type of accumulated gradients. Needs to correspond to the type
  of the accumulator.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Stack")
    .Output("handle: Ref(string)")
    .Attr("elem_type: type")
    .Attr("stack_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
A stack that produces elements in first-in last-out order.

handle: The handle to the stack.
elem_type: The type of the elements on the stack.
stack_name: Overrides the name used for the temporary stack resource. Default
value is the name of the 'Stack' op (which is guaranteed unique).
)doc");

REGISTER_OP("StackPush")
    .Input("handle: Ref(string)")
    .Input("elem: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("swap_memory: bool = false")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Push an element onto the stack.

handle: The handle to a stack.
elem: The tensor to be pushed onto the stack.
output: The same tensor as the input 'elem'.
swap_memory: Swap `elem` to CPU. Default to false.
)doc");

REGISTER_OP("StackPop")
    .Input("handle: Ref(string)")
    .Output("elem: elem_type")
    .Attr("elem_type: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Pop the element at the top of the stack.

handle: The handle to a stack.
elem: The tensor that is popped from the top of the stack.
elem_type: The type of the elem that is popped.
)doc");

REGISTER_OP("StackClose")
    .Input("handle: Ref(string)")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Doc(R"doc(
Delete the stack from its resource container.

handle: The handle to a stack.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("TensorArrayV2")
    .Input("size: int32")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .Attr("dynamic_size: bool = false")
    .Attr("clear_after_read: bool = true")
    .Attr("tensor_array_name: string = ''")
    .Output("handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
    .Doc(R"doc(
An array of Tensors of given size, with data written via Write and read
via Read or Pack.

handle: The handle to the TensorArray.
size: The size of the array.
dtype: The type of the elements on the tensor_array.
element_shape: The expected shape of an element, if known. Used to
  validate the shapes of TensorArray elements. If this shape is not
  fully specified, gathering zero-size TensorArrays is an error.
dynamic_size: A boolean that determines whether writes to the TensorArray
  are allowed to grow the size.  By default, this is not allowed.
clear_after_read: If true (default), Tensors in the TensorArray are cleared
  after being read.  This disables multiple read semantics but allows early
  release of memory.
tensor_array_name: Overrides the name used for the temporary tensor_array
  resource. Default value is the name of the 'TensorArray' op (which
  is guaranteed unique).
)doc");

REGISTER_OP("TensorArrayGradV2")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("grad_handle: string")
    .Attr("source: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
    .Doc(R"doc(
Creates a TensorArray for storing the gradients of values in the given handle.

If the given TensorArray gradient already exists, returns a reference to it.

Locks the size of the original TensorArray by disabling its dynamic size flag.

**A note about the input flow_in:**

The handle flow_in forces the execution of the gradient lookup to occur
only after certain other operations have occurred.  For example, when
the forward TensorArray is dynamically sized, writes to this TensorArray
may resize the object.  The gradient TensorArray is statically sized based
on the size of the forward TensorArray when this operation executes.
Furthermore, the size of the forward TensorArray is frozen by this call.
As a result, the flow is used to ensure that the call to generate the gradient
TensorArray only happens after all writes are executed.

In the case of dynamically sized TensorArrays, gradient computation should
only be performed on read operations that have themselves been chained via
flow to occur only after all writes have executed. That way the final size
of the forward TensorArray is known when this operation is called.

**A note about the source attribute:**

TensorArray gradient calls use an accumulator TensorArray object.  If
multiple gradients are calculated and run in the same session, the multiple
gradient nodes may accidentally flow throuth the same accumulator TensorArray.
This double counts and generally breaks the TensorArray gradient flow.

The solution is to identify which gradient call this particular
TensorArray gradient is being called in.  This is performed by identifying
a unique string (e.g. "gradients", "gradients_1", ...) from the input
gradient Tensor's name.  This string is used as a suffix when creating
the TensorArray gradient object here (the attribute `source`).

The attribute `source` is added as a suffix to the forward TensorArray's
name when performing the creation / lookup, so that each separate gradient
calculation gets its own TensorArray accumulator.

handle: The handle to the forward TensorArray.
flow_in: A float scalar that enforces proper chaining of operations.
source: The gradient source string, used to decide which gradient TensorArray
  to return.
)doc");

REGISTER_OP("TensorArrayWriteV2")
    .Input("handle: string")
    .Input("index: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
Push an element onto the tensor_array.

handle: The handle to a TensorArray.
index: The position to write to inside the TensorArray.
value: The tensor to write to the TensorArray.
flow_in: A float scalar that enforces proper chaining of operations.
flow_out: A float scalar that enforces proper chaining of operations.
)doc");

REGISTER_OP("TensorArrayReadV2")
    .Input("handle: string")
    .Input("index: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::UnknownShape(c);
    })
    .Doc(R"doc(
Read an element from the TensorArray into output `value`.

handle: The handle to a TensorArray.
dtype: The type of the elem that is returned.
flow_in: A float scalar that enforces proper chaining of operations.
value: The tensor that is read from the TensorArray.
)doc");

REGISTER_OP("TensorArrayGatherV2")
    .Input("handle: string")
    .Input("indices: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::UnknownShape(c);
    })
    .Doc(R"doc(
Gather specific elements from the TensorArray into output `value`.

All elements selected by `indices` must have the same shape.

handle: The handle to a TensorArray.
indices: The locations in the TensorArray from which to read tensor elements.
dtype: The type of the elem that is returned.
element_shape: The expected shape of an element, if known. Used to
  validate the shapes of TensorArray elements. If this shape is not
  fully specified, gathering zero-size TensorArrays is an error.
flow_in: A float scalar that enforces proper chaining of operations.
value: All of the elements in the TensorArray, concatenated along a new
  axis (the new dimension 0).
)doc");

REGISTER_OP("TensorArrayScatterV2")
    .Input("handle: string")
    .Input("indices: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
Scatter the data from the input value into specific TensorArray elements.

`indices` must be a vector, its length must match the first dim of `value`.

handle: The handle to a TensorArray.
indices: The locations at which to write the tensor elements.
value: The concatenated tensor to write to the TensorArray.
flow_in: A float scalar that enforces proper chaining of operations.
flow_out: A float scalar that enforces proper chaining of operations.
)doc");

REGISTER_OP("TensorArrayConcatV2")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Output("lengths: int64")
    .Attr("dtype: type")
    .Attr("element_shape_except0: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->UnknownShape());
      c->set_output(1, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Concat the elements from the TensorArray into value `value`.

Takes `T` elements of shapes

  ```
  (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
  ```

and concatenates them into a Tensor of shape:

  ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```

All elements must have the same shape (excepting the first dimension).

handle: The handle to a TensorArray.
dtype: The type of the elem that is returned.
flow_in: A float scalar that enforces proper chaining of operations.
element_shape_except0: The expected shape of an element, if known,
  excluding the first dimension. Used to validate the shapes of
  TensorArray elements. If this shape is not fully specified, concatenating
  zero-size TensorArrays is an error.
value: All of the elements in the TensorArray, concatenated along the first
  axis.
lengths: A vector of the row sizes of the original T elements in the
  value output.  In the example above, this would be the values:
  `(n1, n2, ..., n(T-1))`.
)doc");

REGISTER_OP("TensorArraySplitV2")
    .Input("handle: string")
    .Input("value: T")
    .Input("lengths: int64")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
Split the data from the input value into TensorArray elements.

Assuming that `lengths` takes on values

  ```(n0, n1, ..., n(T-1))```

and that `value` has shape

  ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```,

this splits values into a TensorArray with T tensors.

TensorArray index t will be the subtensor of values with starting position

  ```(n0 + n1 + ... + n(t-1), 0, 0, ...)```

and having size

  ```nt x d0 x d1 x ...```

handle: The handle to a TensorArray.
value: The concatenated tensor to write to the TensorArray.
lengths: The vector of lengths, how to split the rows of value into the
  TensorArray.
flow_in: A float scalar that enforces proper chaining of operations.
flow_out: A float scalar that enforces proper chaining of operations.
)doc");

REGISTER_OP("TensorArraySizeV2")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("size: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
Get the current size of the TensorArray.

handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
flow_in: A float scalar that enforces proper chaining of operations.
size: The current size of the TensorArray.
)doc");

REGISTER_OP("TensorArrayCloseV2")
    .Input("handle: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return Status::OK();
    })
    .Doc(R"doc(
Delete the TensorArray from its resource container.  This enables
the user to close and release the resource in the middle of a step/run.

handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
)doc");

// --------------------------------------------------------------------------

// Deprecated TensorArray methods

REGISTER_OP("TensorArray")
    .Input("size: int32")
    .Attr("dtype: type")
    .Attr("dynamic_size: bool = false")
    .Attr("clear_after_read: bool = true")
    .Attr("tensor_array_name: string = ''")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .Output("handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayV2");
REGISTER_OP("TensorArrayGrad")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("grad_handle: Ref(string)")
    .Attr("source: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayGradV2");
REGISTER_OP("TensorArrayWrite")
    .Input("handle: Ref(string)")
    .Input("index: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayWriteV2");
REGISTER_OP("TensorArrayRead")
    .Input("handle: Ref(string)")
    .Input("index: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayReadV2");
REGISTER_OP("TensorArrayPack")
    .Input("handle: Ref(string)")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayGatherV2 with RangeOp");
REGISTER_OP("TensorArrayUnpack")
    .Input("handle: Ref(string)")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayScatterV2 with RangeOp");
REGISTER_OP("TensorArrayGather")
    .Input("handle: Ref(string)")
    .Input("indices: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayGatherV2");
REGISTER_OP("TensorArrayScatter")
    .Input("handle: Ref(string)")
    .Input("indices: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayScatterV2");
REGISTER_OP("TensorArrayConcat")
    .Input("handle: Ref(string)")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Output("lengths: int64")
    .Attr("dtype: type")
    .Attr("element_shape_except0: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayConcatV2");
REGISTER_OP("TensorArraySplit")
    .Input("handle: Ref(string)")
    .Input("value: T")
    .Input("lengths: int64")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArraySplitV2");
REGISTER_OP("TensorArraySize")
    .Input("handle: Ref(string)")
    .Input("flow_in: float")
    .Output("size: int32")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArraySizeV2");
REGISTER_OP("TensorArrayClose")
    .Input("handle: Ref(string)")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayCloseV2");

// --------------------------------------------------------------------------

REGISTER_OP("Barrier")
    .SetIsStateful()
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
Defines a barrier that persists across different graph executions.

A barrier represents a key-value map, where each key is a string, and
each value is a tuple of tensors.

At runtime, the barrier contains 'complete' and 'incomplete'
elements. A complete element has defined tensors for all components of
its value tuple, and may be accessed using BarrierTakeMany. An
incomplete element has some undefined components in its value tuple,
and may be updated using BarrierInsertMany.

handle: The handle to the barrier.
component_types: The type of each component in a value.
shapes: The shape of each component in a value. Each shape must be 1 in the
  first dimension. The length of this attr must be the same as the length of
  component_types.
capacity: The capacity of the barrier.  The default capacity is MAX_INT32,
  which is the largest capacity of the underlying queue.
container: If non-empty, this barrier is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this barrier will be shared under the given name
  across multiple sessions.
)doc");

REGISTER_OP("BarrierInsertMany")
    .Input("handle: Ref(string)")
    .Input("keys: string")
    .Input("values: T")
    .Attr("T: type")
    .Attr("component_index: int")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle keys = c->input(1);
      ShapeHandle values = c->input(2);
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(keys, 1, &keys));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 1, &values));
      TF_RETURN_IF_ERROR(c->Merge(keys, c->Vector(c->Dim(values, 0)), &handle));
      return Status::OK();
    })
    .Doc(R"doc(
For each key, assigns the respective value to the specified component.

If a key is not found in the barrier, this operation will create a new
incomplete element. If a key is found in the barrier, and the element
already has a value at component_index, this operation will fail with
INVALID_ARGUMENT, and leave the barrier in an undefined state.

handle: The handle to a barrier.
component_index: The component of the barrier elements that is being assigned.
keys: A one-dimensional tensor of keys, with length n.
values: An any-dimensional tensor of values, which are associated with the
  respective keys. The 0th dimension must have length n.
)doc");

REGISTER_OP("BarrierTakeMany")
    .Input("handle: Ref(string)")
    .Input("num_elements: int32")
    .Output("indices: int64")
    .Output("keys: string")
    .Output("values: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("allow_small_batch: bool = false")
    .Attr("wait_for_incomplete: bool = false")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Takes the given number of completed elements from a barrier.

This operation concatenates completed-element component tensors along
the 0th dimension to make a single component tensor.

Elements come out of the barrier when they are complete, and in the order
in which they were placed into the barrier.  The indices output provides
information about the batch in which each element was originally inserted
into the barrier.

handle: The handle to a barrier.
num_elements: A single-element tensor containing the number of elements to
  take.
indices: A one-dimensional tensor of indices, with length num_elems.
  These indices refer to the batch in which the values were placed into the
  barrier (starting with MIN_LONG and increasing with each BarrierInsertMany).
keys: A one-dimensional tensor of keys, with length num_elements.
values: One any-dimensional tensor per component in a barrier element. All
  values have length num_elements in the 0th dimension.
component_types: The type of each component in a value.
allow_small_batch: Allow to return less than num_elements items if barrier is
  already closed.
timeout_ms: If the queue is empty, this operation will block for up to
  timeout_ms milliseconds.
  Note: This option is not supported yet.
)doc");

REGISTER_OP("BarrierClose")
    .Input("handle: Ref(string)")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Attr("cancel_pending_enqueues: bool = false")
    .Doc(R"doc(
Closes the given barrier.

This operation signals that no more new elements will be inserted in the
given barrier. Subsequent InsertMany that try to introduce a new key will fail.
Subsequent InsertMany operations that just add missing components to already
existing elements will continue to succeed. Subsequent TakeMany operations will
continue to succeed if sufficient completed elements remain in the barrier.
Subsequent TakeMany operations that would block will fail immediately.

handle: The handle to a barrier.
cancel_pending_enqueues: If true, all pending enqueue requests that are
  blocked on the barrier's queue will be cancelled. InsertMany will fail, even
  if no new key is introduced.
)doc");

REGISTER_OP("BarrierReadySize")
    .Input("handle: Ref(string)")
    .Output("size: int32")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Doc(R"doc(
Computes the number of complete elements in the given barrier.

handle: The handle to a barrier.
size: The number of complete elements (i.e. those with all of their value
  components set) in the barrier.
)doc");

REGISTER_OP("BarrierIncompleteSize")
    .Input("handle: Ref(string)")
    .Output("size: int32")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Doc(R"doc(
Computes the number of incomplete elements in the given barrier.

handle: The handle to a barrier.
size: The number of incomplete elements (i.e. those with some of their value
  components not set) in the barrier.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("LookupTableFind")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("default_value: Tout")
    .Output("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // Default value must be scalar or vector.
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &unused));
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
Looks up keys in a table, outputs the corresponding values.

The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.

table_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Same shape as `keys`.  Values found in the table, or `default_values`
   for missing keys.
)doc");

REGISTER_OP("LookupTableInsert")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // TODO: Validate keys and values shape.
      return Status::OK();
    })
    .Doc(R"doc(
Updates the table to associates keys with values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

table_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Values to associate with keys.
)doc");

REGISTER_OP("LookupTableSize")
    .Input("table_handle: Ref(string)")
    .Output("size: int64")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Doc(R"doc(
Computes the number of elements in the given table.

table_handle: Handle to the table.
size: Scalar that contains number of elements in the table.
)doc");

REGISTER_OP("LookupTableExport")
    .Input("table_handle: Ref(string)")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 1, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);
      return Status::OK();
    })
    .Doc(R"doc(
Outputs all keys and values in the table.

table_handle: Handle to the table.
keys: Vector of all keys present in the table.
values: Tensor of all values in the table. Indexed in parallel with `keys`.
)doc");

REGISTER_OP("LookupTableImport")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // TODO: Validate keys and values shape.
      return Status::OK();
    })
    .Doc(R"doc(
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

table_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Values to associate with keys.
)doc");

REGISTER_OP("HashTable")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
Creates a non-initialized hash table.

This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.

table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
use_node_name_sharing: If true and shared_name is empty, the table is shared
  using the node name.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

REGISTER_OP("MutableHashTable")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
use_node_name_sharing: If true and shared_name is empty, the table is shared
  using the node name.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

REGISTER_OP("MutableHashTableOfTensors")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

REGISTER_OP("MutableDenseHashTable")
    .Input("empty_key: key_dtype")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .Attr("initial_num_buckets: int = 131072")  // 2^17
    .Attr("max_load_factor: float = 0.8")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
Creates an empty hash table that uses tensors as the backing store. It uses
"open addressing" with quadratic reprobing to resolve collisions.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

empty_key: The key used to represent empty key buckets internally. Must not
  be used in insert or lookup operations.
table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
value_shape: The shape of each value.
initial_num_buckets: The initial number of hash table buckets. Must be a power
  to 2.
max_load_factor: The maximum ratio between number of entries and number of
  buckets before growing the table. Must be between 0 and 1.
)doc");

REGISTER_OP("InitializeTable")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tkey")
    .Input("values: Tval")
    .Attr("Tkey: type")
    .Attr("Tval: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle keys;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
      TF_RETURN_IF_ERROR(c->Merge(keys, c->input(2), &keys));
      return Status::OK();
    })
    .Doc(R"doc(
Table initializer that takes two tensors for keys and values respectively.

table_handle: Handle to a table which will be initialized.
keys: Keys of type Tkey.
values: Values of type Tval.
)doc");

REGISTER_OP("InitializeTableFromTextFile")
    .Input("table_handle: Ref(string)")
    .Input("filename: string")
    .Attr("key_index: int >= -2")
    .Attr("value_index: int >= -2")
    .Attr("vocab_size: int >= -1 = -1")
    .Attr("delimiter: string = '\t'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
      return Status::OK();
    })
    .Doc(R"doc(
Initializes a table from a text file.

It inserts one key-value pair into the table for each line of the file.
The key and value is extracted from the whole line content, elements from the
split line based on `delimiter` or the line number (starting from zero).
Where to extract the key and value from a line is specified by `key_index` and
`value_index`.

- A value of -1 means use the line number(starting from zero), expects `int64`.
- A value of -2 means use the whole line content, expects `string`.
- A value >= 0 means use the index (starting at zero) of the split line based
  on `delimiter`.

table_handle: Handle to a table which will be initialized.
filename: Filename of a vocabulary text file.
key_index: Column index in a line to get the table `key` values from.
value_index: Column index that represents information of a line to get the table
  `value` values from.
vocab_size: Number of elements of the file, use -1 if unknown.
delimiter: Delimiter to separate fields in a line.
)doc");

REGISTER_OP("GetSessionHandle")
    .Input("value: T")
    .Output("handle: string")
    .Attr("T: type")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Store the input tensor in the state of the current session.

value: The tensor to be stored.
handle: The handle for the tensor stored in the session state.
)doc");

REGISTER_OP("GetSessionTensor")
    .Input("handle: string")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      return shape_inference::UnknownShape(c);
    })
    .Doc(R"doc(
Get the value of the tensor specified by its handle.

handle: The handle for a tensor stored in the session state.
value: The tensor for the given handle.
dtype: The type of the output value.
)doc");

REGISTER_OP("DeleteSessionTensor")
    .Input("handle: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      return Status::OK();
    })
    .Doc(R"doc(
Delete the tensor specified by its handle in the session.

handle: The handle for a tensor stored in the session state.
)doc");

}  // namespace tensorflow
