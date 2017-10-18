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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("GatherTree")
    .Input("step_ids: T")
    .Input("parent_ids: T")
    .Input("max_sequence_lengths: int32")
    .Input("end_token: T")
    .Output("beams: T")
    .Attr("T: {int32}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle step_ids, parent_ids, max_sequence_lengths, end_token;

      // step_ids, parent_ids, and output are all shaped:
      //   [max_time, batch_size, beam_width].
      // max_sequence_length is shaped [batch_size] and end_token is a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &step_ids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &parent_ids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &max_sequence_lengths));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &end_token));
      TF_RETURN_IF_ERROR(c->Merge(step_ids, parent_ids, &step_ids));
      DimensionHandle batch_size = c->Dim(step_ids, 1);
      TF_RETURN_IF_ERROR(
          c->Merge(batch_size, c->Dim(max_sequence_lengths, 0), &batch_size));
      ShapeHandle step_ids_prefix = c->Matrix(c->Dim(step_ids, 0), batch_size);
      TF_RETURN_IF_ERROR(c->MergePrefix(step_ids, step_ids_prefix, &step_ids,
                                        &step_ids_prefix));

      c->set_output(0, step_ids);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Calculates the full beams from the per-step ids and parent beam ids.

This op implements the following mathematical equations:

```python
TODO(ebrevdo): fill in
```

step_ids: `[max_time, batch_size, beam_width]`.
parent_ids: `[max_time, batch_size, beam_width]`.
max_sequence_lengths: `[batch_size]`.
end_token: `[]`.
beams: `[max_time, batch_size, beam_width]`.
)doc");

}  // end namespace tensorflow
