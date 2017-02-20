# Custom Data Readers

PREREQUISITES:

*   Some familiarity with C++.
*   Must have
    [downloaded TensorFlow source](../../get_started/os_setup.md#installing-from-sources), and be
    able to build it.

We divide the task of supporting a file format into two pieces:

*   File formats: We use a *Reader* Op to read a *record* (which can be any
    string) from a file.
*   Record formats: We use decoder or parsing Ops to turn a string record
    into tensors usable by TensorFlow.

For example, to read a
[CSV file](https://en.wikipedia.org/wiki/Comma-separated_values), we use
[a Reader for text files](../../api_docs/python/io_ops.md#TextLineReader)
followed by
[an Op that parses CSV data from a line of text](../../api_docs/python/io_ops.md#decode_csv).

[TOC]

## Writing a Reader for a file format

A `Reader` is something that reads records from a file.  There are some examples
of Reader Ops already built into TensorFlow:

*   [`tf.TFRecordReader`](../../api_docs/python/io_ops.md#TFRecordReader)
    ([source in `kernels/tf_record_reader_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/tf_record_reader_op.cc))
*   [`tf.FixedLengthRecordReader`](../../api_docs/python/io_ops.md#FixedLengthRecordReader)
    ([source in `kernels/fixed_length_record_reader_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/fixed_length_record_reader_op.cc))
*   [`tf.TextLineReader`](../../api_docs/python/io_ops.md#TextLineReader)
    ([source in `kernels/text_line_reader_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/text_line_reader_op.cc))

You can see these all expose the same interface, the only differences
are in their constructors.  The most important method is `read`.
It takes a queue argument, which is where it gets filenames to
read from whenever it needs one (e.g. when the `read` op first runs, or
the previous `read` reads the last record from a file).  It produces
two scalar tensors: a string key and a string value.

To create a new reader called `SomeReader`, you will need to:

1.  In C++, define a subclass of
    [`tensorflow::ReaderBase`](https://www.tensorflow.org/code/tensorflow/core/kernels/reader_base.h)
    called `SomeReader`.
2.  In C++, register a new reader op and kernel with the name `"SomeReader"`.
3.  In Python, define a subclass of [`tf.ReaderBase`](https://www.tensorflow.org/code/tensorflow/python/ops/io_ops.py) called `SomeReader`.

You can put all the C++ code in a file in
`tensorflow/core/user_ops/some_reader_op.cc`.  The code to read a file will live
in a descendant of the C++ `ReaderBase` class, which is defined in
[`tensorflow/core/kernels/reader_base.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/reader_base.h).
You will need to implement the following methods:

*   `OnWorkStartedLocked`: open the next file
*   `ReadLocked`: read a record or report EOF/error
*   `OnWorkFinishedLocked`: close the current file, and
*   `ResetLocked`: get a clean slate after, e.g., an error

These methods have names ending in "Locked" since `ReaderBase` makes sure
to acquire a mutex before calling any of these methods, so you generally don't
have to worry about thread safety (though that only protects the members of the
class, not global state).

For `OnWorkStartedLocked`, the name of the file to open is the value returned by
the `current_work()` method.  `ReadLocked` has this signature:

```c++
Status ReadLocked(string* key, string* value, bool* produced, bool* at_end)
```

If `ReadLocked` successfully reads a record from the file, it should fill in:

*   `*key`: with an identifier for the record, that a human could use to find
    this record again.  You can include the filename from `current_work()`,
    and append a record number or whatever.
*   `*value`: with the contents of the record.
*   `*produced`: set to `true`.

If you hit the end of a file (EOF), set `*at_end` to `true`.  In either case,
return `Status::OK()`.  If there is an error, simply return it using one of the
helper functions from
[`tensorflow/core/lib/core/errors.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h)
without modifying any arguments.

Next you will create the actual Reader op.  It will help if you are familiar
with [the adding an op how-to](../../how_tos/adding_an_op/index.md).  The main steps
are:

*   Registering the op.
*   Define and register an `OpKernel`.

To register the op, you will use a `REGISTER_OP` call defined in
[`tensorflow/core/framework/op.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op.h).
Reader ops never take any input and always have a single output with type
`resource`.  They should have string `container` and `shared_name` attrs.
You may optionally define additional attrs
for configuration or include documentation in a `Doc`.  For examples, see
[`tensorflow/core/ops/io_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/ops/io_ops.cc),
e.g.:

```c++
#include "tensorflow/core/framework/op.h"

REGISTER_OP("TextLineReader")
    .Output("reader_handle: resource")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
A Reader that outputs the lines of a file delimited by '\n'.
)doc");
```

To define an `OpKernel`, Readers can use the shortcut of descending from
`ReaderOpKernel`, defined in
[`tensorflow/core/framework/reader_op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/reader_op_kernel.h),
and implement a constructor that calls `SetReaderFactory`.  After defining
your class, you will need to register it using `REGISTER_KERNEL_BUILDER(...)`.
An example with no attrs:

```c++
#include "tensorflow/core/framework/reader_op_kernel.h"

class TFRecordReaderOp : public ReaderOpKernel {
 public:
  explicit TFRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    Env* env = context->env();
    SetReaderFactory([this, env]() { return new TFRecordReader(name(), env); });
  }
};

REGISTER_KERNEL_BUILDER(Name("TFRecordReader").Device(DEVICE_CPU),
                        TFRecordReaderOp);
```

An example with attrs:

```c++
#include "tensorflow/core/framework/reader_op_kernel.h"

class TextLineReaderOp : public ReaderOpKernel {
 public:
  explicit TextLineReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    int skip_header_lines = -1;
    OP_REQUIRES_OK(context,
                   context->GetAttr("skip_header_lines", &skip_header_lines));
    OP_REQUIRES(context, skip_header_lines >= 0,
                errors::InvalidArgument("skip_header_lines must be >= 0 not ",
                                        skip_header_lines));
    Env* env = context->env();
    SetReaderFactory([this, skip_header_lines, env]() {
      return new TextLineReader(name(), skip_header_lines, env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("TextLineReader").Device(DEVICE_CPU),
                        TextLineReaderOp);
```

The last step is to add the Python wrapper.  You can either do this by
[compiling a dynamic
library](../../how_tos/adding_an_op/#building_the_op_library)
or, if you are building TensorFlow from source, adding to `user_ops.py`.
For the latter, you will import `tensorflow.python.ops.io_ops` in
[`tensorflow/python/user_ops/user_ops.py`](https://www.tensorflow.org/code/tensorflow/python/user_ops/user_ops.py)
and add a descendant of [`io_ops.ReaderBase`](https://www.tensorflow.org/code/tensorflow/python/ops/io_ops.py).

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import io_ops

class SomeReader(io_ops.ReaderBase):

    def __init__(self, name=None):
        rr = gen_user_ops.some_reader(name=name)
        super(SomeReader, self).__init__(rr)


ops.NotDifferentiable("SomeReader")
```

You can see some examples in
[`tensorflow/python/ops/io_ops.py`](https://www.tensorflow.org/code/tensorflow/python/ops/io_ops.py).

## Writing an Op for a record format

Generally this is an ordinary op that takes a scalar string record as input, and
so follow [the instructions to add an Op](../../how_tos/adding_an_op/index.md).
You may optionally take a scalar string key as input, and include that in error
messages reporting improperly formatted data.  That way users can more easily
track down where the bad data came from.

Examples of Ops useful for decoding records:

*   [`tf.parse_single_example`](../../api_docs/python/io_ops.md#parse_single_example)
    (and
    [`tf.parse_example`](../../api_docs/python/io_ops.md#parse_example))
*   [`tf.decode_csv`](../../api_docs/python/io_ops.md#decode_csv)
*   [`tf.decode_raw`](../../api_docs/python/io_ops.md#decode_raw)

Note that it can be useful to use multiple Ops to decode a particular record
format.  For example, you may have an image saved as a string in
[a `tf.train.Example` protocol buffer](https://www.tensorflow.org/code/tensorflow/core/example/example.proto).
Depending on the format of that image, you might take the corresponding output
from a
[`tf.parse_single_example`](../../api_docs/python/io_ops.md#parse_single_example)
op and call [`tf.image.decode_jpeg`](../../api_docs/python/image.md#decode_jpeg),
[`tf.image.decode_png`](../../api_docs/python/image.md#decode_png), or
[`tf.decode_raw`](../../api_docs/python/io_ops.md#decode_raw).  It is common to
take the output of `tf.decode_raw` and use
[`tf.slice`](../../api_docs/python/array_ops.md#slice) and
[`tf.reshape`](../../api_docs/python/array_ops.md#reshape) to extract pieces.
