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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// CTC is Connectionist Temporal Classification.  See util/ctc/ for details.

REGISTER_OP("CTCLoss")
    .Input("inputs: T")
    .Input("labels_indices: int64")
    .Input("labels_values: int32")
    .Input("sequence_length: int32")
    .Attr("preprocess_collapse_repeated: bool = false")
    .Attr("ctc_merge_repeated: bool = true")
    .Attr("ignore_longer_outputs_than_inputs: bool = false")
    .Output("loss: T")
    .Output("gradient: T")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle inputs;
      ShapeHandle labels_indices;
      ShapeHandle labels_values;
      ShapeHandle sequence_length;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &labels_indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &labels_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &sequence_length));

      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(labels_indices, 0),
                                  c->Dim(labels_values, 0), &unused));

      // Get batch size from inputs and sequence_length, and update inputs
      // with the merged batch_size since it is returned.
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->ReplaceDim(inputs, 1, batch_size, &inputs));

      c->set_output(0, c->Vector(batch_size));
      c->set_output(1, inputs);
      return Status::OK();
    });

REGISTER_OP("CTCGreedyDecoder")
    .Input("inputs: T")
    .Input("sequence_length: int32")
    .Attr("merge_repeated: bool = false")
    .Output("decoded_indices: int64")
    .Output("decoded_values: int64")
    .Output("decoded_shape: int64")
    .Output("log_probability: T")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle inputs;
      ShapeHandle sequence_length;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));

      // Get batch size from inputs and sequence_length.
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));

      DimensionHandle total_decoded_outputs = c->UnknownDim();
      c->set_output(0, c->Matrix(total_decoded_outputs, 2));
      c->set_output(1, c->Vector(total_decoded_outputs));
      c->set_output(2, c->Vector(2));
      c->set_output(3, c->Matrix(batch_size, 1));
      return Status::OK();
    });

REGISTER_OP("CTCBeamSearchDecoder")
    .Input("inputs: T")
    .Input("sequence_length: int32")
    .Attr("beam_width: int >= 1")
    .Attr("top_paths: int >= 1")
    .Attr("merge_repeated: bool = true")
    .Output("decoded_indices: top_paths * int64")
    .Output("decoded_values: top_paths * int64")
    .Output("decoded_shape: top_paths * int64")
    .Output("log_probability: T")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle inputs;
      ShapeHandle sequence_length;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));

      // Get batch size from inputs and sequence_length.
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));

      int32 top_paths;
      TF_RETURN_IF_ERROR(c->GetAttr("top_paths", &top_paths));

      // Outputs.
      int out_idx = 0;
      for (int i = 0; i < top_paths; ++i) {  // decoded_indices
        c->set_output(out_idx++, c->Matrix(InferenceContext::kUnknownDim, 2));
      }
      for (int i = 0; i < top_paths; ++i) {  // decoded_values
        c->set_output(out_idx++, c->Vector(InferenceContext::kUnknownDim));
      }
      ShapeHandle shape_v = c->Vector(2);
      for (int i = 0; i < top_paths; ++i) {  // decoded_shape
        c->set_output(out_idx++, shape_v);
      }
      c->set_output(out_idx++, c->Matrix(batch_size, top_paths));
      return Status::OK();
    });

}  // namespace tensorflow
