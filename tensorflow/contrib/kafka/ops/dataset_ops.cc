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

REGISTER_OP("KafkaDataset")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that emits the messages of one or more Kafka topics.

topics: A `tf.string` tensor containing one or more subscriptions,
  in the format of [topic:partition:offset:length],
  by default length is -1 for unlimited.
servers: A list of bootstrap servers.
group: The consumer group id.
eof: If True, the kafka reader will stop on EOF.
timeout: The timeout value for the Kafka Consumer to wait
  (in millisecond).
)doc");

}  // namespace tensorflow
