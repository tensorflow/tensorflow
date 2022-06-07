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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status ScalarInputsAndOutputs(InferenceContext* c) {
  ShapeHandle unused;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return OkStatus();
}

Status TwoElementVectorAndScalarOutputs(InferenceContext* c) {
  ShapeHandle handle;
  DimensionHandle unused_handle;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return OkStatus();
}

Status TwoElementOutput(InferenceContext* c) {
  c->set_output(0, c->Vector(2));
  return OkStatus();
}

}  // namespace

REGISTER_OP("SaveV2")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Input("tensors: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate prefix.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names and shapes_and_slices.
      for (int i = 1; i <= 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &s));
        TF_RETURN_IF_ERROR(
            c->WithValue(c->Dim(s, 0), c->num_inputs() - 3, &unused_dim));
      }
      // TODO(mrry): Attempt to parse the shapes_and_slices values and use
      // them to constrain the shape of the remaining inputs.
      return OkStatus();
    });

REGISTER_OP("RestoreV2")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Output("tensors: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle shape0, shape1, shape2;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &shape0));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shape1));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shape2));
      TF_RETURN_IF_ERROR(c->Merge(shape1, shape2, &shape0));

      // Attempt to infer output shapes from its shape_and_slice input.
      const Tensor* shape_and_slices_tensor = c->input_tensor(2);
      if (shape_and_slices_tensor) {
        if (shape_and_slices_tensor->dtype() != DT_STRING) {
          return errors::InvalidArgument(
              "Expected an input tensor of type string.");
        }

        const auto& shape_and_slices_flat =
            shape_and_slices_tensor->flat<tstring>();
        if (shape_and_slices_flat.size() != c->num_outputs()) {
          return errors::InvalidArgument(
              "The number of shape_and_slice doesn't match tensor outputs.");
        }
        for (int i = 0; i < shape_and_slices_flat.size(); ++i) {
          const string& shape_and_slice = shape_and_slices_flat(i);
          if (shape_and_slice.empty()) {
            c->set_output(i, c->UnknownShape());
            continue;
          }
          TensorShape parsed_full_shape;
          TensorSlice parsed_slice;
          TensorShape parsed_slice_shape;
          TF_RETURN_IF_ERROR(checkpoint::ParseShapeAndSlice(
              shape_and_slice, &parsed_full_shape, &parsed_slice,
              &parsed_slice_shape));
          ShapeHandle shape_handle;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromTensorShape(parsed_slice_shape, &shape_handle));
          c->set_output(i, shape_handle);
        }
        return OkStatus();
      } else {
        return UnknownShape(c);
      }
    });

REGISTER_OP("MergeV2Checkpoints")
    .Input("checkpoint_prefixes: string")
    .Input("destination_prefix: string")
    .Attr("delete_old_dirs: bool = true")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return OkStatus();
    });

REGISTER_OP("Save")
    .Input("filename: string")
    .Input("tensor_names: string")
    .Input("data: T")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate filename.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &s));
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(s, 0), c->num_inputs() - 2, &unused_dim));

      return OkStatus();
    });

REGISTER_OP("SaveSlices")
    .Input("filename: string")
    .Input("tensor_names: string")
    .Input("shapes_and_slices: string")
    .Input("data: T")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate filename.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names and unused_shapes_and_slices.
      for (int i = 1; i <= 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &s));
        TF_RETURN_IF_ERROR(
            c->WithValue(c->Dim(s, 0), c->num_inputs() - 3, &unused_dim));
      }
      // TODO(mrry): Attempt to parse the shapes_and_slices values and use
      // them to constrain the shape of the remaining inputs.
      return OkStatus();
    });

REGISTER_OP("Restore")
    .Input("file_pattern: string")
    .Input("tensor_name: string")
    .Output("tensor: dt")
    .Attr("dt: type")
    .Attr("preferred_shard: int = -1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->UnknownShape());
      return OkStatus();
    });

REGISTER_OP("RestoreSlice")
    .Input("file_pattern: string")
    .Input("tensor_name: string")
    .Input("shape_and_slice: string")
    .Output("tensor: dt")
    .Attr("dt: type")
    .Attr("preferred_shard: int = -1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

      // Attempt to infer output shapes from its shape_and_slice input.
      const Tensor* shape_and_slices_tensor = c->input_tensor(2);
      if (shape_and_slices_tensor) {
        const auto& shape_and_slice =
            shape_and_slices_tensor->flat<tstring>()(0);
        if (shape_and_slice.empty()) {
          c->set_output(0, c->UnknownShape());
        } else {
          TensorShape parsed_full_shape;
          TensorSlice parsed_slice;
          TensorShape parsed_slice_shape;
          TF_RETURN_IF_ERROR(checkpoint::ParseShapeAndSlice(
              shape_and_slice, &parsed_full_shape, &parsed_slice,
              &parsed_slice_shape));
          ShapeHandle shape_handle;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromTensorShape(parsed_slice_shape, &shape_handle));
          c->set_output(0, shape_handle);
        }
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("ShardedFilename")
    .Input("basename: string")
    .Input("shard: int32")
    .Input("num_shards: int32")
    .Output("filename: string")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ShardedFilespec")
    .Input("basename: string")
    .Input("num_shards: int32")
    .Output("filename: string")
    .SetShapeFn(ScalarInputsAndOutputs);

// Reader source ops ----------------------------------------------------------

REGISTER_OP("WholeFileReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("WholeFileReaderV2")
    .Output("reader_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TextLineReader")
    .Output("reader_handle: Ref(string)")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use TextLineReaderV2");

REGISTER_OP("TextLineReaderV2")
    .Output("reader_handle: resource")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FixedLengthRecordReader")
    .Output("reader_handle: Ref(string)")
    .Attr("header_bytes: int = 0")
    .Attr("record_bytes: int")
    .Attr("footer_bytes: int = 0")
    .Attr("hop_bytes: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use FixedLengthRecordReaderV2");

REGISTER_OP("FixedLengthRecordReaderV2")
    .Output("reader_handle: resource")
    .Attr("header_bytes: int = 0")
    .Attr("record_bytes: int")
    .Attr("footer_bytes: int = 0")
    .Attr("hop_bytes: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("encoding: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TFRecordReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("compression_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use TFRecordReaderV2");

REGISTER_OP("TFRecordReaderV2")
    .Output("reader_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("compression_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("LMDBReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("IdentityReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use IdentityReaderV2");

REGISTER_OP("IdentityReaderV2")
    .Output("reader_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

// Ops that operate on Readers ------------------------------------------------

REGISTER_OP("ReaderRead")
    .Input("reader_handle: Ref(string)")
    .Input("queue_handle: Ref(string)")
    .Output("key: string")
    .Output("value: string")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderReadV2")
    .Input("reader_handle: resource")
    .Input("queue_handle: resource")
    .Output("key: string")
    .Output("value: string")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ReaderReadUpTo")
    .Input("reader_handle: Ref(string)")
    .Input("queue_handle: Ref(string)")
    .Input("num_records: int64")
    .Output("keys: string")
    .Output("values: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      ShapeHandle out = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, out);
      c->set_output(1, out);
      return OkStatus();
    });

REGISTER_OP("ReaderReadUpToV2")
    .Input("reader_handle: resource")
    .Input("queue_handle: resource")
    .Input("num_records: int64")
    .Output("keys: string")
    .Output("values: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      ShapeHandle out = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, out);
      c->set_output(1, out);
      return OkStatus();
    });

REGISTER_OP("ReaderNumRecordsProduced")
    .Input("reader_handle: Ref(string)")
    .Output("records_produced: int64")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderNumRecordsProducedV2")
    .Input("reader_handle: resource")
    .Output("records_produced: int64")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ReaderNumWorkUnitsCompleted")
    .Input("reader_handle: Ref(string)")
    .Output("units_completed: int64")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderNumWorkUnitsCompletedV2")
    .Input("reader_handle: resource")
    .Output("units_completed: int64")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ReaderSerializeState")
    .Input("reader_handle: Ref(string)")
    .Output("state: string")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderSerializeStateV2")
    .Input("reader_handle: resource")
    .Output("state: string")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ReaderRestoreState")
    .Input("reader_handle: Ref(string)")
    .Input("state: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      DimensionHandle unused_handle;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(c->input(0), 0), 2, &unused_handle));

      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return OkStatus();
    });

REGISTER_OP("ReaderRestoreStateV2")
    .Input("reader_handle: resource")
    .Input("state: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return OkStatus();
    });

REGISTER_OP("ReaderReset")
    .Input("reader_handle: Ref(string)")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderResetV2")
    .Input("reader_handle: resource")
    .SetShapeFn(ScalarInputsAndOutputs);

// Other input Ops ----------------------------------------------------------

REGISTER_OP("ReadFile")
    .Input("filename: string")
    .Output("contents: string")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("WriteFile")
    .Input("filename: string")
    .Input("contents: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return OkStatus();
    });

REGISTER_OP("MatchingFiles")
    .Input("pattern: string")
    .Output("filenames: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      return OkStatus();
    });

}  // namespace tensorflow
