/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

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

REGISTER_OP("SummaryWriter")
    .Output("writer: resource")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("CreateSummaryFileWriter")
    .Input("writer: resource")
    .Input("logdir: string")
    .Input("max_queue: int32")
    .Input("flush_millis: int32")
    .Input("filename_suffix: string")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("CreateSummaryDbWriter")
    .Input("writer: resource")
    .Input("db_uri: string")
    .Input("experiment_name: string")
    .Input("run_name: string")
    .Input("user_name: string")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("FlushSummaryWriter")
    .Input("writer: resource")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("CloseSummaryWriter")
    .Input("writer: resource")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("WriteSummary")
    .Input("writer: resource")
    .Input("step: int64")
    .Input("tensor: T")
    .Input("tag: string")
    .Input("summary_metadata: string")
    .Attr("T: type")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("WriteRawProtoSummary")
    .Input("writer: resource")
    .Input("step: int64")
    .Input("tensor: string")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("ImportEvent")
    .Input("writer: resource")
    .Input("event: string")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("WriteScalarSummary")
    .Input("writer: resource")
    .Input("step: int64")
    .Input("tag: string")
    .Input("value: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("WriteHistogramSummary")
    .Input("writer: resource")
    .Input("step: int64")
    .Input("tag: string")
    .Input("values: T")
    .Attr("T: realnumbertype = DT_FLOAT")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("WriteImageSummary")
    .Input("writer: resource")
    .Input("step: int64")
    .Input("tag: string")
    .Input("tensor: T")
    .Input("bad_color: uint8")
    .Attr("max_images: int >= 1 = 3")
    .Attr("T: {uint8, float, half} = DT_FLOAT")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("WriteAudioSummary")
    .Input("writer: resource")
    .Input("step: int64")
    .Input("tag: string")
    .Input("tensor: float")
    .Input("sample_rate: float")
    .Attr("max_outputs: int >= 1 = 3")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("WriteGraphSummary")
    .Input("writer: resource")
    .Input("step: int64")
    .Input("tensor: string")
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow
