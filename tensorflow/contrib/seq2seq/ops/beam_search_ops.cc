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
    .Input("sequence_length: T")
    .Output("beams: T")
    .Attr("T: {int32}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle step_ids, parent_ids, sequence_length;

      // step_ids, parent_ids, and output are all shaped:
      //   [max_time, batch_size, beam_width].
      // sequence_length is shaped [batch_size, beam_width].
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &step_ids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &parent_ids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &sequence_length));

      DimensionHandle batch_size = c->Dim(step_ids, 1);
      DimensionHandle beam_width = c->Dim(step_ids, 2);

      TF_RETURN_IF_ERROR(c->Merge(step_ids, parent_ids, &step_ids));
      TF_RETURN_IF_ERROR(
          c->Merge(batch_size, c->Dim(sequence_length, 0), &batch_size));
      TF_RETURN_IF_ERROR(
          c->Merge(beam_width, c->Dim(sequence_length, 1), &beam_width));

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
sequence_length: `[batch_size, beam_width]`.
beams: `[max_time, batch_size, beam_width]`.
)doc");

}  // end namespace tensorflow
