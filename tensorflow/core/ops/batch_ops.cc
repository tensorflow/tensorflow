/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

REGISTER_OP("BatchFunction")
    .Input("in_tensors: Tin")
    .Input("captured_tensors: Tcaptured")
    .Output("out_tensors: Tout")
    .Attr("f: func")
    .Attr("num_batch_threads: int")
    // 'max_batch_size' denotes the maximum batch size acceptable, i.e., inputs
    // with larger batch size are simply invalidated.
    // By default, 'max_batch_size' must be equal to max value of
    // 'allowed_batch_sizes'.
    // By setting 'enable_large_batch_splitting' (attribute below) to true,
    // 'max_batch_size' can be greater than or equal to max value of
    // 'allowed_batch_sizes', in other words,
    // 1) input with size > 'max_batch_size' is still invalidated.
    // 2) input with
    //    a) size <= 'max_batch_size'
    //    b) size > max value of 'allowed_batch_sizes'
    //    will automatically be split into multiple batches (with batch size in
    //    'allowed_batch_sizes'), executed, and re-composed (as final output).
    .Attr("max_batch_size: int")
    .Attr("batch_timeout_micros: int")
    .Attr("max_enqueued_batches: int = 10")
    .Attr("allowed_batch_sizes: list(int) = []")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("batching_queue: string = ''")
    .Attr("Tin: list(type)")
    .Attr("Tcaptured: list(type) >= 0")
    .Attr("Tout: list(type)")
    // If 'enable_large_batch_splitting' is true, for input batches exceeding
    // the largest value in "allowed_batch_sizes", allow the batch to be split
    // into multiple batches with batch size within "allowed_batch_sizes".
    // NOTE: Support for `enable_large_batch_splitting == true` is still
    // developed in progress.
    .Attr("enable_large_batch_splitting: bool = false")
    // TODO(apassos): Fix this shape inference function. It requires shape
    // inference of function calls.
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Batch")
    .Input("in_tensors: T")
    .Output("batched_tensors: T")
    .Output("batch_index: int64")
    .Output("id: int64")
    .Attr("num_batch_threads: int")
    .Attr("max_batch_size: int")
    .Attr("max_enqueued_batches: int = 10")
    .Attr("batch_timeout_micros: int")
    .Attr("allowed_batch_sizes: list(int) = []")
    .Attr("grad_timeout_micros: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("batching_queue: string = ''")
    .Attr("T: list(type)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<shape_inference::ShapeHandle> in_shapes;
      TF_RETURN_IF_ERROR(c->input("in_tensors", &in_shapes));
      std::vector<shape_inference::ShapeHandle> out_shapes(in_shapes.size());
      for (int i = 0; i < in_shapes.size(); ++i) {
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(in_shapes[i], 0, c->UnknownDim(), &out_shapes[i]));
      }
      TF_RETURN_IF_ERROR(c->set_output("batched_tensors", out_shapes));
      TF_RETURN_IF_ERROR(c->set_output("id", {c->Scalar()}));
      TF_RETURN_IF_ERROR(c->set_output(
          "batch_index",
          {c->MakeShape({shape_inference::DimensionOrConstant(c->UnknownDim()),
                         shape_inference::DimensionOrConstant(3)})}));
      return Status::OK();
    });

REGISTER_OP("Unbatch")
    .Input("batched_tensor: T")
    .Input("batch_index: int64")
    .Input("id: int64")
    .Output("unbatched_tensor: T")
    .Attr("timeout_micros: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle out_shape;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &out_shape));
      c->set_output(0, out_shape);
      return Status::OK();
    });

REGISTER_OP("UnbatchGrad")
    .Input("original_input: T")
    .Input("batch_index: int64")
    .Input("grad: T")
    .Input("id: int64")
    .Output("batched_grad: T")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(c->Rank(c->input(2))));
      return Status::OK();
    });

}  // namespace tensorflow
