/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

// --------------------------------------------------------------------------

REGISTER_OP("DynamicPartition")
    .Input("data: T")
    .Input("partitions: int32")
    .Output("outputs: num_partitions * T")
    .Attr("num_partitions: int")
    .Attr("T: type")
    .Doc(R"doc(
Partitions `data` into `num_partitions` tensors using indices from `partitions`.

For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
are placed in `outputs[i]` in lexicographic order of `js`, and the first
dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
In detail,

    outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

    outputs[i] = pack([data[js, ...] for js if partitions[js] == i])

`data.shape` must start with `partitions.shape`.

For example:

    # Scalar partitions
    partitions = 1
    num_partitions = 2
    data = [10, 20]
    outputs[0] = []  # Empty with shape [0, 2]
    outputs[1] = [[10, 20]]

    # Vector partitions
    partitions = [0, 0, 1, 1, 0]
    num_partitions = 2
    data = [10, 20, 30, 40, 50]
    outputs[0] = [10, 20, 50]
    outputs[1] = [30, 40]

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
    .Attr("N : int >= 2")
    .Attr("T : type")
    .Doc(R"doc(
Interleave the values from the `data` tensors into a single tensor.

Builds a merged tensor such that

    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]

For example, if each `indices[m]` is scalar or vector, we have

    # Scalar indices
    merged[indices[m], ...] = data[m][...]

    # Vector indices
    merged[indices[m][i], ...] = data[m][i, ...]

Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is

    merged.shape = [max(indices)] + constant

Values are merged in order, so if an index appears in both `indices[m][i]` and
`indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
merged result.

For example:

    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/DynamicStitch.png" alt>
</div>
)doc");

// --------------------------------------------------------------------------

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

REGISTER_OP("QueueEnqueue")
    .Input("handle: Ref(string)")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
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
    .Doc(R"doc(
Dequeues n tuples of one or more tensors from the given queue.

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

REGISTER_OP("QueueClose")
    .Input("handle: Ref(string)")
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
    .Doc(R"doc(
Computes the number of elements in the given queue.

handle: The handle to a queue.
size: The number of elements in the given queue.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Stack")
    .Output("handle: Ref(string)")
    .Attr("elem_type: type")
    .Attr("stack_name: string = ''")
    .SetIsStateful()
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
    .Doc(R"doc(
Push an element onto the stack.

handle: The handle to a stack.
elem: The tensor to be pushed onto the stack.
output: The same tensor as the input 'elem'.
)doc");

REGISTER_OP("StackPop")
    .Input("handle: Ref(string)")
    .Output("elem: elem_type")
    .Attr("elem_type: type")
    .Doc(R"doc(
Pop the element at the top of the stack.

handle: The handle to a stack.
elem_type: The type of the elem that is popped.
elem: The tensor that is popped from the top of the stack.
)doc");

REGISTER_OP("StackClose")
    .Input("handle: Ref(string)")
    .Doc(R"doc(
Delete the stack from its resource container.

handle: The handle to a stack.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("LookupTableFind")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("default_value: Tout")
    .Output("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
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

REGISTER_OP("LookupTableSize")
    .Input("table_handle: Ref(string)")
    .Output("size: int64")
    .Doc(R"doc(
Computes the number of elements in the given table.

table_handle: Handle to the table.
size: Scalar that contains number of elements in the table.
)doc");

REGISTER_OP("HashTable")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
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
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

REGISTER_OP("InitializeTable")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tkey")
    .Input("values: Tval")
    .Attr("Tkey: type")
    .Attr("Tval: type")
    .Doc(R"doc(
Table initializer that takes two tensors for keys and values respectively.

table_handle: Handle to a table which will be initialized.
keys: Keys of type Tkey.
values: Values of type Tval. Same shape as `keys`.
)doc");

}  // namespace tensorflow
