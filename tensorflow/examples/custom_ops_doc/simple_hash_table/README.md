<!-- LINT.IfChange -->
# Create a stateful custom op

This example shows you how to create TensorFlow custom ops with internal state.
These custom ops use resources to hold their state. You do this by implementing
a stateful C++ data structure instead of using tensors to hold the state. This
example also covers:

*   Using `tf.Variable`s for state as an alternative to using a resource (which
    is recommended for all cases where storing the state in fixed size tensors
    is reasonable)
*   Saving and restoring the state of the custom op by using `SavedModel`s
*   The `IsStateful` flag which is true for ops with state and other unrelated
    cases
*   A set of related custom ops and ops with various signatures (number of
    inputs and outputs)

For additional context,
read the
[OSS guide on creating custom ops](https://www.tensorflow.org/guide/create_op).

## Background

### Overview of resources and ref-counting

TensorFlow C++ resource classes are derived from the
[`ResourceBase` class](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_base.h).
A resource is referenced in the Tensorflow Graph as a
[`ResourceHandle`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_handle.h)
Tensor, of the type
[`DT_RESOURCE`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto).
A resource can be owned by a ref-counting `ResourceHandle`, or by a
[`ResourceMgr`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_mgr.h)
(deprecated for new ops).

A handle-owned resource using ref-counting is automatically destroyed when all
resource handles pointing to it go out of scope. If the resource needs to be
looked up from a name, for example, if the resource handle is serialized and
deserialized, the resource must be published as an unowned entry with
[`ResourceMgr::CreateUnowned`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_mgr.h#L641)
upon creation. The entry is a weak reference to the resource, and is
automatically removed when the resource goes out of scope.

In contrast, with the deprecated `ResourceMgr` owned resources, a resource
handle behaves like a weak ref - it does not destroy the underlying resource
when its lifetime ends.
[`DestroyResourceOp`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/resource_variable_ops.cc#L339)
must be explicitly called to destroy the resource. An example of why requiring
calling `DestroyResourceOp` is problematic is that it is easy to introduce a bug
when adding a new code path to an op that returns without calling
`DestroyResourceOp`.

While it would be possible to have a data structure that has a single operation
implemented by a single custom op (for example, something that just has a 'next
state' operation that requires no initialization), typically a related set of
custom ops are used with a resource. **Separating creation and use into
different custom ops is recommended.** Typically, one custom op creates the
resource and one or more additional custom ops implement functionality to access
or modify it.

### Using `tf.Variable`s for state

An alternative to custom ops with internal state is to store the state
externally in one or more
[`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)s as
tensors (as detailed [here](https://www.tensorflow.org/guide/variable)) and have
one or more (normal, stateless) ops that use tensors stored in these
`tf.Variable`s. One example is `tf.random.Generator`.

For cases where using variables is possible and efficient, using them is
preferred since the implementation does not require adding a new C++ resource
type. An example of a case where using variables is not possible and custom ops
using a resource must be used is where the amount of space for data grows
dynamically.

Here is a toy example of using the
`multiplex_2`
custom op with a `tf.Variable` in the same manner as an in-built TensorFlow op
is used with variables. The variable is initialized to 1 for every value.
Indices from a
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
cause the corresponding element of the variable to be doubled.

<!-- test_snippets_in_readme skip -->
```python
import tensorflow as tf
from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op

def variable_and_stateless_op():
  n = 10
  v = tf.Variable(tf.ones(n, dtype=tf.int64), trainable=False)
  dataset = tf.data.Dataset.from_tensor_slices([5, 1, 7, 5])
  for position in dataset:
    print(v.numpy())
    cond = tf.one_hot(
        position, depth=n, on_value=True, off_value=False, dtype=bool)
    v.assign(multiplex_2_op.multiplex(cond, v*2, v))
  print(v.numpy())
```

This outputs:

<!-- test_snippets_in_readme skip -->
```
[1 1 1 1 1 1 1 1 1 1]
[1 1 1 1 1 2 1 1 1 1]
[1 2 1 1 1 2 1 1 1 1]
[1 2 1 1 1 2 1 2 1 1]
[1 2 1 1 1 4 1 2 1 1]
```

It is also possible to pass a Python `tf.Variable` handle as a
[`DT_RESOURCE`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto)
input to a custom op where its C++ kernel uses the handle to access the variable
using the variable's internal C++ API. This is not a common case because the
variable's C++ API provides little extra functionality. It can be appropriate
for cases that only sparsely update the variable.

### The `IsStateful` flag

All TensorFlow ops have an
[`IsStateful`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_mgr.h#L505)
flag. It is set to `True` for ops that have internal state, both for ops that
follow the recommended pattern of using a resource and those that have internal
state in some other way. In addition, it is also set to `True` for ops with side
effects (for example, I/O) and for disabling certain optimizations for an op.

### Save and restore state with `SavedModel`

A [`SavedModel`](https://www.tensorflow.org/guide/saved_model) contains a
complete TensorFlow program, including trained parameters (like `tf.Variable`s)
and computation. In some cases, you can save and restore the state in a custom
op by using `SavedModel`. This is optional, but is important for some cases, for
example if a custom op is used during training and the state needs to persist
when training is restarted from a checkpoint. The hash table in this example
supports `SavedModel` with a custom Python wrapper that implements the
`_serialize_to_tensors` and `_restore_from_tensors` methods of
[`tf.saved_model.experimental.TrackableResource`](https://www.tensorflow.org/api_docs/python/tf/saved_model/experimental/TrackableResource)
and ops that are used by these methods.

## Creating a hash table with a stateful custom op

This example demonstrates how to implement custom ops that use ref-counting
resources to track state. This example creates a `simple_hash_table` which is
similar to
[`tf.lookup.experimental.MutableHashTable`](https://www.tensorflow.org/api_docs/python/tf/lookup/experimental/MutableHashTable).

The hash table implements a set of 4 CRUD (Create/Read/Update/Delete) style
custom ops. Each of these ops has a different signature and they illustrate
cases such as custom ops with more than one output.

The hash table supports `SavedModel` through the `export` and `import` ops to
save/restore state. For an actual hash table use case, it is preferable to use
(or extend) the existing
[`tf.lookup`](https://www.tensorflow.org/api_docs/python/tf/lookup) ops. In this
simple example, `insert`, `find`, and `remove` use only a single key-value pair
per call. In contrast, existing `tf.lookup` ops can use multiple key-value
pairs in a single call.

The table below summarizes the six custom ops implemented in this example.

Operation | Purpose                     | Resource class method | Kernel class                    | Custom op                      | Python class member
--------- | --------------------------- | --------------------- | ------------------------------- | ------------------------------ | -------------------
create    | CRUD and SavedModel: create | Default constructor   | `SimpleHashTableCreateOpKernel` | Examples>SimpleHashTableCreate | `__init__`
find      | CRUD: read                  | Find                  | `SimpleHashTableFindOpKernel`   | Examples>SimpleHashTableFind   | `find`
insert    | CRUD: update                | Insert                | `SimpleHashTableInsertOpKernel` | Examples>SimpleHashTableInsert | `insert`
remove    | CRUD: delete                | Remove                | `SimpleHashTableRemoveOpKernel` | Examples>SimpleHashTableRemove | `remove`
import    | SavedModel: restore         | Import                | `SimpleHashTableImportOpKernel` | Examples>SimpleHashTableImport | `do_import`
export    | SavedModel: save            | Export                | `SimpleHashTableExportOpKernel` | Examples>SimpleHashTableExport | `export`

You can use this hash table as:

<!-- test_snippets_in_readme skip -->
```python
hash_table = simple_hash_table_op.SimpleHashTable(tf.int32, float,
                                                  default_value=-999.0)
result1 = hash_table.find(key=1, dynamic_default_value=-999.0)
# -999.0
hash_table.insert(key=1, value=100.0)
result2 = hash_table.find(key=1, dynamic_default_value=-999.0)
# 100.0
```

The example below contains C++ and Python code snippets to illustrate the code
flow. These snippets are not all complete; some are missing namespace
declarations, imports, and test cases.

This example deviates slightly from the general recipe for creating TensorFlow
custom ops. The most significant differences are noted in each step below.

### Step 0 - Implement the resource class

Implement a
`SimpleHashTableResource`
resource class derived from
[`ResourceBase`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_base.h).

```live-snippet
template <class K, class V>
class SimpleHashTableResource : public ::tensorflow::ResourceBase {
 public:
  Status Insert(const Tensor& key, const Tensor& value) {
    const K key_val = key.flat<K>()(0);
    const V value_val = value.flat<V>()(0);

    mutex_lock l(mu_);
    table_[key_val] = value_val;
    return OkStatus();
  }

  Status Find(const Tensor& key, Tensor* value, const Tensor& default_value) {
    // Note that tf_shared_lock could be used instead of mutex_lock
    // in ops that do not not modify data protected by a mutex, but
    // go/totw/197 recommends using exclusive lock instead of a shared
    // lock when the lock is not going to be held for a significant amount
    // of time.
    mutex_lock l(mu_);

    const V default_val = default_value.flat<V>()(0);
    const K key_val = key.flat<K>()(0);
    auto value_val = value->flat<V>();
    value_val(0) = gtl::FindWithDefault(table_, key_val, default_val);
    return OkStatus();
  }

  Status Remove(const Tensor& key) {
    mutex_lock l(mu_);

    const K key_val = key.flat<K>()(0);
    if (table_.erase(key_val) != 1) {
      return errors::NotFound("Key for remove not found: ", key_val);
    }
    return OkStatus();
  }

  // Save all key, value pairs to tensor outputs to support SavedModel
  Status Export(OpKernelContext* ctx) {
    mutex_lock l(mu_);
    int64_t size = table_.size();
    Tensor* keys;
    Tensor* values;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({size}), &keys));
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("values", TensorShape({size}), &values));
    auto keys_data = keys->flat<K>();
    auto values_data = values->flat<V>();
    int64_t i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      keys_data(i) = it->first;
      values_data(i) = it->second;
    }
    return OkStatus();
  }

  // Load all key, value pairs from tensor inputs to support SavedModel
  Status Import(const Tensor& keys, const Tensor& values) {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();

    mutex_lock l(mu_);
    table_.clear();
    for (int64_t i = 0; i < key_values.size(); ++i) {
      gtl::InsertOrUpdate(&table_, key_values(i), value_values(i));
    }
    return OkStatus();
  }

  // Create a debug string with the content of the map if this is small,
  // or some example data if this is large, handling both the cases where the
  // hash table has many entries and where the entries are long strings.
  std::string DebugString() const override { return DebugString(3); }
  std::string DebugString(int num_pairs) const {
    std::string rval = "SimpleHashTable {";
    size_t count = 0;
    const size_t max_kv_str_len = 100;
    mutex_lock l(mu_);
    for (const auto& pair : table_) {
      if (count >= num_pairs) {
        strings::StrAppend(&rval, "...");
        break;
      }
      std::string kv_str = strings::StrCat(pair.first, ": ", pair.second);
      strings::StrAppend(&rval, kv_str.substr(0, max_kv_str_len));
      if (kv_str.length() > max_kv_str_len) strings::StrAppend(&rval, " ...");
      strings::StrAppend(&rval, ", ");
      count += 1;
    }
    strings::StrAppend(&rval, "}");
    return rval;
  }

 private:
  mutable mutex mu_;
  absl::flat_hash_map<K, V> table_ TF_GUARDED_BY(mu_);
};
```

Note that this class provides:

*   Helper methods to access the hash table. These methods correspond to the
    `find`, `insert`, and `remove` ops.
*   Helper methods to `import`/`export` the complete internal state of the hash
    table. These methods help support `SavedModel`.
*   A [mutex](https://en.wikipedia.org/wiki/Lock_\(computer_science\)) for the
    helper methods to use for exclusive access to the
    [`absl::flat_hash_map`](https://abseil.io/docs/cpp/guides/container#abslflat_hash_map-and-abslflat_hash_set).
    This ensures thread safety by ensuring that only one thread at a time can
    access the data in the hash table.

### Step 1 - Define the op interface

Define op interfaces and register all the custom ops you create for the hash
table. You typically define one custom op to create the resource. You also
define one or more custom ops that correspond to operations on the data
structure. You also define custom ops to perform import and export operations
that input/output the whole internal state. These ops are optional; you define
them in this example to support `SavedModel`. As the resource is automatically
deleted based on ref-counting, there is no custom op required to delete the
resource.

The `simple_hash_table` has kernels for the `create`, `insert`, `find`,
`remove`, `import`, and `export` ops which use the resource object that actually
stores and manipulates data. The resource is passed between ops using a resource
handle. The `create` op has an `Output` of type `resource`. The other ops have
an `Input` of type `resource`. The interface definitions for all the ops along
with their shape functions are in
`simple_hash_table_op.cc`.

Note that these definitions do not explicitly use `SetIsStateful`. The
`IsStateful` flag is set automatically for any op with an input or output of
type `DT_RESOURCE`.

The definitions below and their corresponding kernels and generated Python
wrappers illustrate the following cases:

*   0, 1, 2, and 3 inputs
*   0, 1, and 2 outputs
*   `Attr` for types where none are used by `Input` or `Output` and all have to
    be explicitly passed into the generated wrapper (for example,
    `SimpleHashTableCreate`)
*   `Attr` for types where all are used by `Input` and/or `Output` (so they are
    set implicitly inside the generated wrapper) and none are passed into the
    generated wrapper (for example, `SimpleHashTableFind`)
*   `Attr` for types where some but not all are used by `Input` or `Output` and
    only those that are not used are explicitly passed into the generated
    wrapper (for example, `SimpleHashTableRemove` where there is an `Input` that
    uses `key_dtype` but the `value_type` `Attr` is a parameter to the generated
    wrapper).

```
REGISTER_OP("Examples>SimpleHashTableCreate")
    .Output("output: resource")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ScalarOutput);
```
```
REGISTER_OP("Examples>SimpleHashTableFind")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Input("default_value: value_dtype")
    .Output("value: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ThreeScalarInputsScalarOutput);
```
```
REGISTER_OP("Examples>SimpleHashTableInsert")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Input("value: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ThreeScalarInputs);
```
```
REGISTER_OP("Examples>SimpleHashTableRemove")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(TwoScalarInputs);
```
```
REGISTER_OP("Examples>SimpleHashTableExport")
    .Input("table_handle: resource")
    .Output("keys: key_dtype")
    .Output("values: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ExportShapeFunction);
```
```
REGISTER_OP("Examples>SimpleHashTableImport")
    .Input("table_handle: resource")
    .Input("keys: key_dtype")
    .Input("values: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ImportShapeFunction);
```

### Step 2 - Register the op implementation (kernel)

Declare kernels for specific types of key-value pairs. Register the kernel by
calling the `REGISTER_KERNEL_BUILDER` macro.

```
#define REGISTER_KERNEL(key_dtype, value_dtype)               \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableCreate")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableCreateOpKernel<key_dtype, value_dtype>); \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableFind")                    \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableFindOpKernel<key_dtype, value_dtype>);   \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableInsert")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableInsertOpKernel<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableRemove")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableRemoveOpKernel<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableExport")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableExportOpKernel<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableImport")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableImportOpKernel<key_dtype, value_dtype>);
```
```
REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int32, tstring);
REGISTER_KERNEL(int64_t, double);
REGISTER_KERNEL(int64_t, float);
REGISTER_KERNEL(int64_t, int32);
REGISTER_KERNEL(int64_t, int64_t);
REGISTER_KERNEL(int64_t, tstring);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64_t);
REGISTER_KERNEL(tstring, tstring);
```

### Step 3 - Implement the op kernel(s)

The implementation of kernels for the ops in the hash table all use helper
functions from the `SimpleHashTableResource` resource class.

You implement the op kernels in two phases:

1.  Implement the kernel for the `create` op that has a `Compute` method that
    creates a resource object and a ref-counted handle for it.

    <!-- test_snippets_in_readme skip -->
    ```c++
    handle_tensor.scalar<ResourceHandle>()() =
        ResourceHandle::MakeRefCountingHandle(
            new SimpleHashTableResource<K, V>(), /* … */);
    ```

1.  Implement the kernels(s) for each of the other operations on the resource
    that have Compute methods that get a resource object and use one or more of
    its helper methods.

    <!-- test_snippets_in_readme skip -->
    ```c++
    MyResource* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    // The GetResource local function uses handle.GetResource<resource_type>()
    OP_REQUIRES_OK(ctx, resource->Find(key, out, default_value));
    ```

#### Creating the resource

The `create` op creates a resource object of type `SimpleHashTableResource` and
then uses `MakeRefCountingHandle` to pass the ownership to a resource handle.
This op outputs a `resource` handle.

```
template <class K, class V>
class SimpleHashTableCreateOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableCreateOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor handle_tensor;
    AllocatorAttributes attr;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                           &handle_tensor, attr));
    handle_tensor.scalar<ResourceHandle>()() =
        ResourceHandle::MakeRefCountingHandle(
            new SimpleHashTableResource<K, V>(), ctx->device()->name(),
            /*dtypes_and_shapes=*/{}, ctx->stack_trace());
    ctx->set_output(0, handle_tensor);
  }

 private:
  // Just to be safe, avoid accidentally copying the kernel.
  TF_DISALLOW_COPY_AND_ASSIGN(SimpleHashTableCreateOpKernel);
};
```

#### Getting the resource

In
`simple_hash_table_kernel.cc`,
the `GetResource` helper function uses an input `resource` handle to retrieve
the corresponding `SimpleHashTableResource` object. It is used by all the custom
ops that use the resource (that is, all the custom ops in the set other than
`create`).

```
template <class K, class V>
Status GetResource(OpKernelContext* ctx,
                   SimpleHashTableResource<K, V>** resource) {
  const Tensor& handle_tensor = ctx->input(0);
  const ResourceHandle& handle = handle_tensor.scalar<ResourceHandle>()();
  typedef SimpleHashTableResource<K, V> resource_type;
  TF_ASSIGN_OR_RETURN(*resource, handle.GetResource<resource_type>());
  return OkStatus();
}
```

#### Using the resource

The ops that use the resource use `GetResource` to get a pointer to the resource
object and call the corresponding helper function for that object. Below is the
source code for the `find` op which uses `resource->Find`. The other ops that
use the resource similarly use the corresponding helper method in the resource
class.

```
template <class K, class V>
class SimpleHashTableFindOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableFindOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v(),
                                      DataTypeToEnum<V>::v()};
    DataTypeVector expected_outputs = {DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    // Note that ctx->input(0) is the Resource handle
    const Tensor& key = ctx->input(1);
    const Tensor& default_value = ctx->input(2);
    TensorShape output_shape = default_value.shape();
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("value", output_shape, &out));
    OP_REQUIRES_OK(ctx, resource->Find(key, out, default_value));
  }
};
```

#### Compile the op

Compile the C++ op to create a kernel library and Python wrapper that enables
you to use the op with TensorFlow.

Create a `BUILD` file for the op which declares the dependencies and the output
build targets. Refer to
[building for OSS](https://www.tensorflow.org/guide/create_op#build_the_op_library).

Note
that you will be reusing this `BUILD` file later on in this example.

```
tf_custom_op_library(
    name = "simple_hash_table_kernel.so",
    srcs = [
        "simple_hash_table_kernel.cc",
        "simple_hash_table_op.cc",
    ],
    deps = [
        "//third_party/absl/container:flat_hash_map",
        "//third_party/tensorflow/core/lib/gtl:map_util",
        "//third_party/tensorflow/core/platform:strcat",
    ],
)

py_strict_library(
    name = "simple_hash_table_op",
    srcs = ["simple_hash_table_op.py"],
    data = ["simple_hash_table_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_strict_library(
    name = "simple_hash_table",
    srcs = ["simple_hash_table.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_hash_table_op",
        "//third_party/py/tensorflow",
    ],
)
```

### Step 4 - Create the Python wrapper

To create the Python wrapper, import and implement a function that serves as the
op's public API and provides a docstring.

The Python wrapper for this example, `simple_hash_table.py` uses the
`SimpleHashTable` class to provide methods that allow access to the data
structure. The custom ops are used to implement these methods. This class also
supports `SavedModel` which allows the state of the resource to be saved and
restored for checkpointing. The class is derived from the
`tf.saved_model.experimental.TrackableResource` base class and implements the
`_serialize_to_tensors` (which uses the `export` op) and `_restore_from_tensors`
(which uses the `import` op) methods defined by this base class. The `__init__`
method of this class calls the `_create_resource` method (which is an override
of a base class method required for `SavedModel`) which in turn calls the C++
`examples_simple_hash_table_create` kernel. The resource handle returned by this
kernel is stored in the private `self._resource_handle` object member. Methods
that use this handle access it using the public `self.resource_handle` property
provided by the `tf.saved_model.experimental.TrackableResource` base class.

```
  def _create_resource(self):
    """Create the resource tensor handle.

    `_create_resource` is an override of a method in base class
    `TrackableResource` that is required for SavedModel support. It can be
    called by the `resource_handle` property defined by `TrackableResource`.

    Returns:
      A tensor handle to the lookup table.
    """
    assert self._default_value.get_shape().ndims == 0
    table_ref = gen_simple_hash_table_op.examples_simple_hash_table_create(
        key_dtype=self._key_dtype,
        value_dtype=self._value_dtype,
        name=self._name)
    return table_ref

  def _serialize_to_tensors(self):
    """Implements checkpointing protocols for `Trackable`."""
    tensors = self.export()
    return {"table-keys": tensors[0], "table-values": tensors[1]}

  def _restore_from_tensors(self, restored_tensors):
    """Implements checkpointing protocols for `Trackable`."""
    return gen_simple_hash_table_op.examples_simple_hash_table_import(
        self.resource_handle, restored_tensors["table-keys"],
        restored_tensors["table-values"])
```

The `find` method in this class calls the `examples_simple_hash_table_find`
custom op using the reference handle (from the public `self.resource_handle`
property) and a key and default value. The other methods are similar. It is
recommended that methods for using a resource avoid logic other than a call to a
generated Python wrapper to avoid eager/`tf.function` inconsistencies (avoid
Python logic that is lost when a `tf.function` is created). In the case of
`find`, using `tf.convert_to_tensor` cannot be avoided and is not lost during
`tf.function` creation.

```
  def find(self, key, dynamic_default_value=None, name=None):
    """Looks up `key` in a table, outputs the corresponding value.

    The `default_value` is used if key not present in the table.

    Args:
      key: Key to look up. Must match the table's key_dtype.
      dynamic_default_value: The value to use if the key is missing in the
        table. If None (by default), the `table.default_value` will be used.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the value in the same shape as `key` using the
        table's value type.

    Raises:
      TypeError: when `key` do not match the table data types.
    """
    with tf.name_scope(name or "%s_lookup_table_find" % self._name):
      key = tf.convert_to_tensor(key, dtype=self._key_dtype, name="key")
      if dynamic_default_value is not None:
        dynamic_default_value = tf.convert_to_tensor(
            dynamic_default_value,
            dtype=self._value_dtype,
            name="default_value")
      value = gen_simple_hash_table_op.examples_simple_hash_table_find(
          self.resource_handle, key, dynamic_default_value
          if dynamic_default_value is not None else self._default_value)
    return value
```

The Python wrapper specifies that gradients are not implemented in this example.
For an example of a differentiable map. For general information about gradients,
read
[Implement the gradient in Python](https://www.tensorflow.org/guide/create_op#implement_the_gradient_in_python)
in the OSS guide.

```
tf.no_gradient("Examples>SimpleHashTableCreate")
tf.no_gradient("Examples>SimpleHashTableFind")
tf.no_gradient("Examples>SimpleHashTableInsert")
tf.no_gradient("Examples>SimpleHashTableRemove")
```

The full source code for the Python wrapper is in
`simple_hash_table_op.py`]
and
`simple_hash_table.py`.

### Step 5 - Test the op

Create op tests using classes derived from
[`tf.test.TestCase`](https://www.tensorflow.org/api_docs/python/tf/test/TestCase)
and the
[parameterized tests provided by Abseil](https://github.com/abseil/abseil-py/blob/main/absl/testing/parameterized.py).

Create tests using three or more custom ops to create `SimpleHashTable` objects
and:

1.  Update the state using the `insert`, `remove`, and/or `import` methods
1.  Observe the state using the `find` and/or `export` methods

Here is an example test:

```
  def test_find_insert_find_strings_eager(self):
    default = 'Default'
    foo = 'Foo'
    bar = 'Bar'
    hash_table = simple_hash_table.SimpleHashTable(tf.string, tf.string,
                                                   default)
    result1 = hash_table.find(foo, default)
    self.assertEqual(result1, default)
    hash_table.insert(foo, bar)
    result2 = hash_table.find(foo, default)
    self.assertEqual(result2, bar)
```

Create a helper function to work with the hash table:

```
  def _use_table(self, key_dtype, value_dtype):
    hash_table = simple_hash_table.SimpleHashTable(key_dtype, value_dtype, 111)
    result1 = hash_table.find(1, -999)
    hash_table.insert(1, 100)
    result2 = hash_table.find(1, -999)
    hash_table.remove(1)
    result3 = hash_table.find(1, -999)
    results = tf.stack((result1, result2, result3))
    return results  # expect [-999, 100, -999]
```

The following test explicitly creates a `tf.function` with `_use_table`.
Ref-counting causes the C++ resource object created in `_use_table` to be
destroyed when this `tf.function` returns inside `self.evaluate(results)`. By
explicitly creating the `tf.function` instead of relying on decorators like
`@test_util.with_eager_op_as_function` and
`@test_util.run_in_graph_and_eager_modes` (such as in
`multiplex_1`),
you have an explicit place in the test corresponding to where the C++ resource's
destructor is called.

This test also shows a test that is parameterized for different data types; it
is actually two tests, one with `tf.int32` input / `float` output and the other
with `tf.int64` input / `tf.int32` output.

```
  def test_find_insert_find_tf_function(self, key_dtype, value_dtype):
    results = def_function.function(
        lambda: self._use_table(key_dtype, value_dtype))
    self.assertAllClose(self.evaluate(results), [-999.0, 100.0, -999.0])
```

Reuse the `BUILD` file to add build
rules for the Python API wrapper and the op test.

```
tf_custom_op_library(
    name = "simple_hash_table_kernel.so",
    srcs = [
        "simple_hash_table_kernel.cc",
        "simple_hash_table_op.cc",
    ],
    deps = [
        "//third_party/absl/container:flat_hash_map",
        "//third_party/tensorflow/core/lib/gtl:map_util",
        "//third_party/tensorflow/core/platform:strcat",
    ],
)

py_strict_library(
    name = "simple_hash_table_op",
    srcs = ["simple_hash_table_op.py"],
    data = ["simple_hash_table_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_strict_library(
    name = "simple_hash_table",
    srcs = ["simple_hash_table.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_hash_table_op",
        "//third_party/py/tensorflow",
    ],
)

tf_py_test(
    name = "simple_hash_table_test",
    size = "medium",  # This test blocks because it writes and reads a file,
    timeout = "short",  # but it still runs quickly.
    srcs = ["simple_hash_table_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":simple_hash_table",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python/framework:errors",
        "//third_party/tensorflow/python/framework:test_lib",
    ],
)
```

Test the op by running:

<!-- test_snippets_in_readme skip -->
```shell
$ bazel test //third_party/tensorflow/google/g3doc/example/simple_hash_table:simple_hash_table_test
```

### Use the op

Use the op by importing and calling it as follows:

<!-- test_snippets_in_readme skip -->
```python
import tensorflow as tf

from tensorflow.examples.custom_ops_doc.simple_hash_table import simple_hash_table

hash_table = simple_hash_table.SimpleHashTable(tf.int32, float, -999.0)
result1 = hash_table.find(1, -999.0)  # -999.0
hash_table.insert(1, 100.0)
result2 = hash_table.find(1, -999.0)  # 100.0
```

Here, `simple_hash_table` is the name of the Python wrapper that was created
in this example.

### Summary

In this example, you learned how to implement a simple hash table data structure
using stateful custom ops.

The table below summarizes the build rules and targets for building and testing
the `simple_hash_table` op.

Op components                           | Build rule             | Build target               | Source
--------------------------------------- | ---------------------- | -------------------------- | ------
Kernels (C++)                           | `tf_custom_op_library` | `simple_hash_table_kernel` | `simple_hash_table_kernel.cc`, `simple_hash_table_op.cc`
Wrapper (automatically generated)       | N/A.                   | `gen_simple_hash_table_op` | N/A
Wrapper (with public API and docstring) | `py_strict_library`    | `simple_hash_table_op`, `simple_hash_table`     | `simple_hash_table_op.py`, `simple_hash_table.py`
Tests                                   | `tf_py_test`           | `simple_hash_table_test`   | `simple_hash_table_test.py`
<!-- LINT.ThenChange(simple_hash_table.md) -->

