// See docs in ../ops/io_ops.cc
#include <unordered_map>

#include "tensorflow/core/kernels/io.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

namespace {
bool ParseShapeAndSlice(const string& shape_and_slice, TensorShape* shape,
                        TensorSlice* slice, TensorShape* shape_slice,
                        string* error) {
  CHECK(!shape_and_slice.empty());
  // Syntax: dim0 dim1 dim2 ... <slice string>
  // Where slice string is defined in core/framework/tensor_slice.h
  std::vector<string> splits = str_util::Split(shape_and_slice, ' ');

  // Must have at least 2 strings.
  if (splits.size() < 2) {
    *error = strings::StrCat(
        "Need least two elements in shape_and_slice specification: ",
        shape_and_slice);
    return false;
  }
  int num_dims = splits.size() - 1;
  shape->Clear();
  for (int i = 0; i < num_dims; ++i) {
    int dim;
    if (!str_util::NumericParse32(splits[i], &dim)) {
      *error = strings::StrCat("Non numerical dimension in shape_and_slice: ",
                               shape_and_slice);
      return false;
    }
    shape->AddDim(dim);
  }
  // The last split is the slice specification.
  slice->Clear();
  auto status = slice->Parse(splits.back(), slice);
  if (!status.ok()) {
    *error = status.error_message();
    return false;
  }
  // The specified slice must be compatible with the specified shape.
  status = slice->SliceTensorShape(*shape, shape_slice);
  if (!status.ok()) {
    *error = status.error_message();
    return false;
  }
  return true;
}
}  // namespace

void SaveTensors(
    OpKernelContext* context,
    checkpoint::TensorSliceWriter::CreateBuilderFunction builder_func,
    bool save_slices) {
  const Tensor& filename_t = context->input(0);
  {
    const int64 size = filename_t.NumElements();
    OP_REQUIRES(
        context, size == 1,
        errors::InvalidArgument(
            "Input 0 (filename) must be a string scalar; got a tensor of ",
            size, "elements"));
  }

  const Tensor& tensor_names_t = context->input(1);
  const int64 N = tensor_names_t.NumElements();
  const string* tensor_shapes_and_slices_ptr = nullptr;
  if (save_slices) {
    const Tensor& tensor_shapes_and_slices_t = context->input(2);
    OP_REQUIRES(
        context, tensor_shapes_and_slices_t.NumElements() == N,
        errors::InvalidArgument("Expected ", N,
                                " elements for the tensor "
                                "shapes and slices but got ",
                                tensor_shapes_and_slices_t.NumElements()));
    tensor_shapes_and_slices_ptr =
        tensor_shapes_and_slices_t.flat<string>().data();
  }
  // Path, names, and slices if save_slices is true.
  const int kFixedInputs = save_slices ? 3 : 2;
  OP_REQUIRES(context, context->num_inputs() == N + kFixedInputs,
              errors::InvalidArgument("Expected totally ", N + kFixedInputs,
                                      " inputs as input #1 (which is a string "
                                      "tensor of saved names) contains ",
                                      N, " names, but received ",
                                      context->num_inputs(), " inputs"));

  VLOG(1) << "About to save tensors to file " << filename_t.flat<string>()(0)
          << "...";
  checkpoint::TensorSliceWriter writer(filename_t.flat<string>()(0),
                                       builder_func);

  Status s;
  auto tensor_names_flat = tensor_names_t.flat<string>();

  string error;
  for (int64 i = 0; i < N; ++i) {
    const string& name = tensor_names_flat(i);
    const Tensor& input = context->input(i + kFixedInputs);
    TensorShape shape(input.shape());
    TensorSlice slice(input.dims());
    if (save_slices && !tensor_shapes_and_slices_ptr[i].empty()) {
      const string& shape_spec = tensor_shapes_and_slices_ptr[i];
      TensorShape slice_shape;
      OP_REQUIRES(context, ParseShapeAndSlice(shape_spec, &shape, &slice,
                                              &slice_shape, &error),
                  errors::InvalidArgument(error));
      OP_REQUIRES(context, slice_shape.IsSameSize(input.shape()),
                  errors::InvalidArgument("Slice in shape_and_slice "
                                          "specification does not match the "
                                          "shape of the tensor to  save: ",
                                          shape_spec, ", tensor: ",
                                          input.shape().DebugString()));
    }

#define WRITER_ADD(dt)                                             \
  case dt:                                                         \
    s = writer.Add(name, shape, slice,                             \
                   input.flat<EnumToDataType<dt>::Type>().data()); \
    break

    switch (input.dtype()) {
      WRITER_ADD(DT_FLOAT);
      WRITER_ADD(DT_DOUBLE);
      WRITER_ADD(DT_INT32);
      WRITER_ADD(DT_UINT8);
      WRITER_ADD(DT_INT16);
      WRITER_ADD(DT_INT8);
      WRITER_ADD(DT_INT64);
      WRITER_ADD(DT_QUINT8);
      WRITER_ADD(DT_QINT8);
      WRITER_ADD(DT_QINT32);
      default:
        context->SetStatus(errors::Unimplemented("Saving data type ",
                                                 DataTypeString(input.dtype()),
                                                 " not yet supported"));
        return;
    }
#undef WRITER_ADD
    if (!s.ok()) {
      context->SetStatus(s);
      return;
    }
  }

  s = writer.Finish();
  if (!s.ok()) {
    context->SetStatus(s);
  }
}

void RestoreTensor(OpKernelContext* context,
                   checkpoint::TensorSliceReader::OpenTableFunction open_func,
                   int preferred_shard, bool restore_slice) {
  const Tensor& file_pattern_t = context->input(0);
  {
    const int64 size = file_pattern_t.NumElements();
    OP_REQUIRES(
        context, size == 1,
        errors::InvalidArgument(
            "Input 0 (file_pattern) must be a string scalar; got a tensor of ",
            size, "elements"));
  }
  const string& file_pattern = file_pattern_t.flat<string>()(0);

  const Tensor& tensor_name_t = context->input(1);
  {
    const int64 size = tensor_name_t.NumElements();
    OP_REQUIRES(
        context, size == 1,
        errors::InvalidArgument(
            "Input 1 (tensor_name) must be a string scalar; got a tensor of ",
            size, "elements"));
  }
  const string& tensor_name = tensor_name_t.flat<string>()(0);

  const string* tensor_shape_and_slice_ptr = nullptr;
  if (restore_slice) {
    const Tensor& tensor_shape_and_slice_t = context->input(2);
    OP_REQUIRES(
        context, tensor_shape_and_slice_t.NumElements() == 1,
        errors::InvalidArgument("Expected 1 element for the tensor "
                                "shape and slice but got ",
                                tensor_shape_and_slice_t.NumElements()));
    tensor_shape_and_slice_ptr = tensor_shape_and_slice_t.flat<string>().data();
  }

  // If we cannot find a cached reader we will allocate our own.
  std::unique_ptr<checkpoint::TensorSliceReader> allocated_reader;

  const checkpoint::TensorSliceReader* reader =
      context->slice_reader_cache()->GetReader(file_pattern, open_func,
                                               preferred_shard);
  if (!reader) {
    allocated_reader.reset(new checkpoint::TensorSliceReader(
        file_pattern, open_func, preferred_shard));
    reader = allocated_reader.get();
  }
  OP_REQUIRES_OK(context, CHECK_NOTNULL(reader)->status());

  // Get the shape and type from the save file.
  DataType type;
  TensorShape saved_shape;
  OP_REQUIRES(
      context, reader->HasTensor(tensor_name, &saved_shape, &type),
      errors::NotFound("Tensor name \"", tensor_name,
                       "\" not found in checkpoint files ", file_pattern));
  OP_REQUIRES(
      context, type == context->expected_output_dtype(0),
      errors::InvalidArgument("Expected to restore a tensor of type ",
                              DataTypeString(context->expected_output_dtype(0)),
                              ", got a tensor of type ", DataTypeString(type),
                              " instead: tensor_name = ", tensor_name));

  // Shape of the output and slice to load.
  TensorShape output_shape(saved_shape);
  TensorSlice slice_to_load(saved_shape.dims());
  if (restore_slice && !tensor_shape_and_slice_ptr[0].empty()) {
    const string& shape_spec = tensor_shape_and_slice_ptr[0];
    TensorShape parsed_shape;
    string error;
    OP_REQUIRES(context,
                ParseShapeAndSlice(shape_spec, &parsed_shape, &slice_to_load,
                                   &output_shape, &error),
                errors::InvalidArgument(error));
    OP_REQUIRES(
        context, parsed_shape.IsSameSize(saved_shape),
        errors::InvalidArgument(
            "Shape in shape_and_slice spec does not match the shape in the "
            "save file: ",
            parsed_shape.DebugString(), ", save file shape: ",
            saved_shape.DebugString()));
  }

  Tensor* t = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &t));
#define READER_COPY(dt)                                                \
  case dt:                                                             \
    reader->CopySliceData(tensor_name, slice_to_load,                  \
                          t->flat<EnumToDataType<dt>::Type>().data()); \
    break

  switch (type) {
    READER_COPY(DT_FLOAT);
    READER_COPY(DT_DOUBLE);
    READER_COPY(DT_INT32);
    READER_COPY(DT_UINT8);
    READER_COPY(DT_INT16);
    READER_COPY(DT_INT8);
    READER_COPY(DT_INT64);
    default:
      context->SetStatus(errors::Unimplemented(
          "Restoring data type ", DataTypeString(type), " not yet supported"));
  }
}

}  // namespace tensorflow
