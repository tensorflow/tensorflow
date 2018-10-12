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
#include "tensorflow/core/framework/dataset_stateful_op_whitelist.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("Assert")
    .Input("condition: bool")
    .Input("data: T")
    .SetIsStateful()
    .Attr("T: list(type)")
    .Attr("summarize: int = 3")
    .SetShapeFn(shape_inference::NoOutputs);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("Assert");

REGISTER_OP("Print")
    .Input("input: T")
    .Input("data: U")
    .Output("output: T")
    .SetIsStateful()
    .Attr("T: type")
    .Attr("U: list(type) >= 0")
    .Attr("message: string = ''")
    .Attr("first_n: int = -1")
    .Attr("summarize: int = 3")
    .SetShapeFn(shape_inference::UnchangedShape);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("Print");

REGISTER_OP("PrintV2")
    .Input("input: string")
    .SetIsStateful()
    .Attr(
        "output_stream: {'stdout', 'stderr', 'log(info)', "
        "'log(warning)', 'log(error)'} = 'stderr'")
    .SetShapeFn([](InferenceContext* c) {
      // Make sure that the input is a scalar.
      if (c->Rank(c->input(0)) != 0) {
        return errors::InvalidArgument("input must be a scalar, but has rank: ",
                                       c->Rank(c->input(0)));
      }
      return Status::OK();
    });

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("PrintV2");

// ----------------------------------------------------------------------------
// Operators that deal with SummaryProtos (encoded as DT_STRING tensors) as
// inputs or outputs in various ways.

REGISTER_OP("TensorSummaryV2")
    .Input("tag: string")
    .Input("tensor: T")
    // This serialized summary metadata field describes a summary value,
    // specifically which plugins may use that summary.
    .Input("serialized_summary_metadata: string")
    .Output("summary: string")
    .Attr("T: type")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TensorSummary")
    .Input("tensor: T")
    .Output("summary: string")
    .Attr("T: type")
    .Attr("description: string = ''")
    .Attr("labels: list(string) = []")
    .Attr("display_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ScalarSummary")
    .Input("tags: string")
    .Input("values: T")
    .Output("summary: string")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("HistogramSummary")
    .Input("tag: string")
    .Input("values: T")
    .Output("summary: string")
    .Attr("T: realnumbertype = DT_FLOAT")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ImageSummary")
    .Input("tag: string")
    .Input("tensor: T")
    .Output("summary: string")
    .Attr("max_images: int >= 1 = 3")
    .Attr("T: {uint8, float, half, float64} = DT_FLOAT")
    .Attr(
        "bad_color: tensor = { dtype: DT_UINT8 "
        "tensor_shape: { dim { size: 4 } } "
        "int_val: 255 int_val: 0 int_val: 0 int_val: 255 }")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("AudioSummaryV2")
    .Input("tag: string")
    .Input("tensor: float")
    .Input("sample_rate: float")
    .Output("summary: string")
    .Attr("max_outputs: int >= 1 = 3")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("AudioSummary")
    .Input("tag: string")
    .Input("tensor: float")
    .Output("summary: string")
    .Attr("sample_rate: float")
    .Attr("max_outputs: int >= 1 = 3")
    .SetShapeFn(shape_inference::ScalarShape)
    .Deprecated(15, "Use AudioSummaryV2.");

REGISTER_OP("MergeSummary")
    .Input("inputs: N * string")
    .Output("summary: string")
    .Attr("N : int >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("Timestamp")
    .Output("ts: float64")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("Timestamp");

}  // end namespace tensorflow
