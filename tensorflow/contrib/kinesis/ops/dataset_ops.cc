/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("KinesisDataset")
    .Input("stream: string")
    .Input("shard: string")
    .Input("read_indefinitely: bool")
    .Input("interval: int64")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that emits the messages of one or more Kinesis topics.

stream: A `tf.string` tensor containing the name of the stream.
shard: A `tf.string` tensor containing the id of the shard.
read_indefinitely: If `True`, the Kinesis dataset will keep retry
  again on `EOF` after the `interval` period. If `False`, then
  the dataset will stop on `EOF`. The default value is `True`.
interval: The interval for the Kinesis Client to wait before
  it tries to get records again (in millisecond).
)doc");

}  // namespace tensorflow
