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
    // A separate set of batch options for the low priority requests, which is
    // used for priority queue batching.
    .Attr("low_priority_max_batch_size: int = 0")
    .Attr("low_priority_batch_timeout_micros: int = 0")
    .Attr("low_priority_allowed_batch_sizes: list(int) = []")
    .Attr("low_priority_max_enqueued_batches: int = 0")
    // Policy that determines the mixed priority batching behavior when low
    // priority batch parameters are present.
    //
    // low_priority_padding_with_next_allowed_batch_size: If high priority
    // batches time out without reaching the max batch size, low priority inputs
    // pad the high priority batches up to the next allowed batch size. A low
    // priority only batch gets schedule only when the low priority input times
    // out or reaches the max batch size while there is no high priority input
    // waiting to be processed.
    // low_priority_padding_with_max_batch_size: Same as above but pad up to the
    // max batch size.
    // priority_isolation: High priority and low priority inputs never share the
    // same batch, i.e., no low priority input padding high priority batches.
    // Low priority inputs get scheduled only as part of low priority only
    // batches as described above.
    // priority_merge: High and low priority inputs are queued separately but
    // when a batch needs to be scheduled, the two queues are treated as one
    // merged flat list of inputs with high priority inputs at the front of the
    // list of tasks to use for the next batch. If all inputs are of the same
    // priority, the behavior is the same as disabling prioritization.
    .Attr(
        "mixed_priority_policy: "
        "{'low_priority_padding_with_max_batch_size', "
        "'low_priority_padding_with_next_allowed_batch_size', "
        "'priority_isolation', 'priority_merge'} = "
        "'low_priority_padding_with_max_batch_size'")
    // The policy that a batch scheduler is using when deciding what to do when,
    // say, 18 requests need to be batched, but only 16 and 32 batch sizes are
    // allowed. The following options are available.
    //
    //   - PAD_UP: pad to size 32.
    //   - BATCH_DOWN: schedule a batch of size 16 and leave 2 requests in the
    //     batch buffer.
    //   - MINIMIZE_TPU_COST_PER_REQUEST: a smarter greedy policy that chooses
    //     to either PAD_UP or BATCH_DOWN so as to minimize the TPU costs per
    //     real request. In this case, it would compare (batch_16_cost / 16) and
    //     (batch_32_cost / 18).
    //
    // WARNING: Not all batch schedulers might support this attribute.
    .Attr(
        "batch_padding_policy: "
        "{'PAD_UP', 'BATCH_DOWN', 'MINIMIZE_TPU_COST_PER_REQUEST'} = 'PAD_UP'")
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
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsDistributedCommunication();

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
      return absl::OkStatus();
    })
    .SetIsDistributedCommunication();

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
      return absl::OkStatus();
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
      return absl::OkStatus();
    });

}  // namespace tensorflow
