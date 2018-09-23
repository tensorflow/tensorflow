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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("PackedSequenceAlignment")
    .Input("sequence_lengths: T")
    .Output("alignments: T")
    .Output("batch_sizes: T")
    .Attr("T: {int8, int16, int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
	  auto alignments_shape = c->MakeShape({
		  c->UnknownDim()
	  });
	  auto batch_sizes_shape = c->MakeShape({
		  c->UnknownDim()
	  });
      c->set_output(0, alignments_shape);
      c->set_output(1, batch_sizes_shape);
      return Status::OK();
    });
	
REGISTER_OP("SequenceGatherScatterIndices")
    .Input("total_length: T")
    .Input("sequence_lengths: T")
    .Input("batch_order: T")
    .Output("gather_scatter_indices: T")
    .Attr("T: {int8, int16, int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
	  auto gather_scatter_indices_shape = c->MakeShape({
		  c->UnknownDim(), 2
	  });
      c->set_output(0, gather_scatter_indices_shape);
      return Status::OK();
    });
	
REGISTER_OP("PackSequence")
    .Input("sequence: T")
    .Input("alignments: Index")
    .Input("batch_sizes: Index")
    .Output("packed_sequence: T")
    .Attr("T: {int32, int64, float32, float64}")
    .Attr("Index: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
	  auto sequence_shape = c->input(0);
	  auto packed_sequence_shape = c->MakeShape({
		  c->UnknownDim(),
		  c->Dim(sequence_shape, 2)
	  });
      c->set_output(0, packed_sequence_shape);
      return Status::OK();
    });
	
REGISTER_OP("UnpackSequence")
    .Input("packed: T")
    .Input("alignments: Index")
    .Input("batch_sizes: Index")
    .Output("sequence: T")
    .Attr("T: {int32, int64, float32, float64}")
    .Attr("Index: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
	  auto packed_shape = c->input(0);
	  auto alignments_shape = c->input(1);
	  auto sequence_shape = c->MakeShape({
		  c->Dim(alignments_shape,0),
		  c->UnknownDim(),
		  c->Dim(packed_shape, 1)
	  });
      c->set_output(0, sequence_shape);
      return Status::OK();
    });

}
}  // namespace tensorflow
