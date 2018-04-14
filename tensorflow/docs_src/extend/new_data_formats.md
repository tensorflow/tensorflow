# Reading custom file and record formats

PREREQUISITES:

*   Some familiarity with C++.
*   Must have
    @{$install_sources$downloaded TensorFlow source}, and be
    able to build it.

We divide the task of supporting a file format into two pieces:

*   File formats: We use a reader `tf.data.Dataset` to read raw *records* (which
    are typically represented by scalar string tensors, but can have more
    structure) from a file.
*   Record formats: We use decoder or parsing ops to turn a string record
    into tensors usable by TensorFlow.

For example, to read a
[CSV file](https://en.wikipedia.org/wiki/Comma-separated_values), we use
@{tf.data.TextLineDataset$a dataset for reading text files line-by-line}
and then @{tf.data.Dataset.map$map} an
@{tf.decode_csv$op} that parses CSV data from each line of text in the dataset.

[TOC]

## Writing a `Dataset` for a file format

A @{tf.data.Dataset} represents a sequence of *elements*, which can be the
individual records in a file. There are several examples of "reader" datasets
that are already built into TensorFlow:

*   @{tf.data.TFRecordDataset}
    ([source in `kernels/data/reader_dataset_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/data/reader_dataset_ops.cc))
*   @{tf.data.FixedLengthRecordDataset}
    ([source in `kernels/data/reader_dataset_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/data/reader_dataset_ops.cc))
*   @{tf.data.TextLineDataset}
    ([source in `kernels/data/reader_dataset_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/data/reader_dataset_ops.cc))

Each of these implementations comprises three related classes:

* A `tensorflow::DatasetOpKernel` subclass (e.g. `TextLineDatasetOp`), which
  tells TensorFlow how to construct a dataset object from the inputs to and
  attrs of an op, in its `MakeDataset()` method.

* A `tensorflow::GraphDatasetBase` subclass (e.g. `TextLineDatasetOp::Dataset`),
  which represents the *immutable* definition of the dataset itself, and tells
  TensorFlow how to construct an iterator object over that dataset, in its
  `MakeIterator()` method.

* A `tensorflow::DatasetIterator<Dataset>` subclass (e.g.
  `TextLineDatasetOp::Dataset::Iterator`), which represents the *mutable* state
  of an iterator over a particular dataset, and tells TensorFlow how to get the
  next element from the iterator, in its `GetNextInternal()` method.

The most important method is the `GetNextInternal()` method, since it defines
how to actually read records from the file and represent them as one or more
`Tensor` objects.

To create a new reader dataset called (for example) `MyReaderDataset`, you will
need to:

1. In C++, define subclasses of `tensorflow::DatasetOpKernel`,
   `tensorflow::GraphDatasetBase`, and `tensorflow::DatasetIterator<Dataset>`
   that implement the reading logic.
2. In C++, register a new reader op and kernel with the name
   `"MyReaderDataset"`.
3. In Python, define a subclass of @{tf.data.Dataset} called `MyReaderDataset`.

You can put all the C++ code in a single file, such as
`my_reader_dataset_op.cc`. It will help if you are
familiar with @{$adding_an_op$the adding an op how-to}. The following skeleton
can be used as a starting point for your implementation:

```c++
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace {

class MyReaderDatasetOp : public DatasetOpKernel {
 public:

  MyReaderDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    // Parse and validate any attrs that define the dataset using
    // `ctx->GetAttr()`, and store them in member variables.
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    // Parse and validate any input tensors 0that define the dataset using
    // `ctx->input()` or the utility function
    // `ParseScalarArgument<T>(ctx, &arg)`.

    // Create the dataset object, passing any (already-validated) arguments from
    // attrs or input tensors.
    *output = new Dataset(ctx);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx) : GraphDatasetBase(ctx) {}

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::MyReader")}));
    }

    // Record structure: Each record is represented by a scalar string tensor.
    //
    // Dataset elements can have a fixed number of components of different
    // types and shapes; replace the following two methods to customize this
    // aspect of the dataset.
    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() override { return "MyReaderDatasetOp::Dataset"; }

   protected:
    // Optional: Implementation of `GraphDef` serialization for this dataset.
    //
    // Implement this method if you want to be able to save and restore
    // instances of this dataset (and any iterators over it).
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              Node** output) const override {
      // Construct nodes to represent any of the input tensors from this
      // object's member variables using `b->AddScalar()` and `b->AddVector()`.
      std::vector<Node*> input_tensors;
      TF_RETURN_IF_ERROR(b->AddDataset(this, input_tensors, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params), i_(0) {}

      // Implementation of the reading logic.
      //
      // The example implementation in this file yields the string "MyReader!"
      // ten times. In general there are three cases:
      //
      // 1. If an element is successfully read, store it as one or more tensors
      //    in `*out_tensors`, set `*end_of_sequence = false` and return
      //    `Status::OK()`.
      // 2. If the end of input is reached, set `*end_of_sequence = true` and
      //    return `Status::OK()`.
      // 3. If an error occurs, return an error status using one of the helper
      //    functions from "tensorflow/core/lib/core/errors.h".
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // NOTE: `GetNextInternal()` may be called concurrently, so it is
        // recommended that you protect the iterator state with a mutex.
        mutex_lock l(mu_);
        if (i_ < 10) {
          // Create a scalar string tensor and add it to the output.
          Tensor record_tensor(ctx->allocator({}), DT_STRING, {});
          record_tensor.scalar<string>()() = "MyReader!";
          out_tensors->emplace_back(std::move(record_tensor));
          ++i_;
          *end_of_sequence = false;
        } else {
          *end_of_sequence = true;
        }
        return Status::OK();
      }

     protected:
      // Optional: Implementation of iterator state serialization for this
      // iterator.
      //
      // Implement these two methods if you want to be able to save and restore
      // instances of this iterator.
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("i"), i_));
        return Status::OK();
      }
      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("i"), &i_));
        return Status::OK();
      }

     private:
      mutex mu_;
      int64 i_ GUARDED_BY(mu_);
    };
  };
};

// Register the op definition for MyReaderDataset.
//
// Dataset ops always have a single output, of type `variant`, which represents
// the constructed `Dataset` object.
//
// Add any attrs and input tensors that define the dataset here.
REGISTER_OP("MyReaderDataset")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

// Register the kernel implementation for MyReaderDataset.
REGISTER_KERNEL_BUILDER(Name("MyReaderDataset").Device(DEVICE_CPU),
                        MyReaderDatasetOp);

}  // namespace
}  // namespace tensorflow
```

The last step is to build the C++ code and add a Python wrapper. The easiest way
to do this is by @{$adding_an_op#build_the_op_library$compiling a dynamic
library} (e.g. called `"my_reader_dataset_op.so"`), and adding a Python class
that subclasses @{tf.data.Dataset} to wrap it. An example Python program is
given here:

```python
import tensorflow as tf

# Assumes the file is in the current working directory.
my_reader_dataset_module = tf.load_op_library("./my_reader_dataset_op.so")

class MyReaderDataset(tf.data.Dataset):

  def __init__(self):
    super(MyReaderDataset, self).__init__()
    # Create any input attrs or tensors as members of this class.

  def _as_variant_tensor(self):
    # Actually construct the graph node for the dataset op.
    #
    # This method will be invoked when you create an iterator on this dataset
    # or a dataset derived from it.
    return my_reader_dataset_module.my_reader_dataset()

  # The following properties define the structure of each element: a scalar
  # `tf.string` tensor. Change these properties to match the `output_dtypes()`
  # and `output_shapes()` methods of `MyReaderDataset::Dataset` if you modify
  # the structure of each element.
  @property
  def output_types(self):
    return tf.string

  @property
  def output_shapes(self):
    return tf.TensorShape([])

  @property
  def output_classes(self):
    return tf.Tensor

if __name__ == "__main__":
  # Create a MyReaderDataset and print its elements.
  with tf.Session() as sess:
    iterator = MyReaderDataset().make_one_shot_iterator()
    next_element = iterator.get_next()
    try:
      while True:
        print(sess.run(next_element))  # Prints "MyReader!" ten times.
    except tf.errors.OutOfRangeError:
      pass
```

You can see some examples of `Dataset` wrapper classes in
[`tensorflow/python/data/ops/dataset_ops.py`](https://www.tensorflow.org/code/tensorflow/python/data/ops/dataset_ops.py).

## Writing an Op for a record format

Generally this is an ordinary op that takes a scalar string record as input, and
so follow @{$adding_an_op$the instructions to add an Op}.
You may optionally take a scalar string key as input, and include that in error
messages reporting improperly formatted data.  That way users can more easily
track down where the bad data came from.

Examples of Ops useful for decoding records:

*   @{tf.parse_single_example} (and @{tf.parse_example})
*   @{tf.decode_csv}
*   @{tf.decode_raw}

Note that it can be useful to use multiple Ops to decode a particular record
format.  For example, you may have an image saved as a string in
[a `tf.train.Example` protocol buffer](https://www.tensorflow.org/code/tensorflow/core/example/example.proto).
Depending on the format of that image, you might take the corresponding output
from a @{tf.parse_single_example} op and call @{tf.image.decode_jpeg},
@{tf.image.decode_png}, or @{tf.decode_raw}.  It is common to take the output
of `tf.decode_raw` and use @{tf.slice} and @{tf.reshape} to extract pieces.
