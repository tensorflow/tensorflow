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

namespace tensorflow {

REGISTER_OP("KafkaDataset")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("servers: string")
    .Attr("group: string")
    .Attr("eof: bool")
    .Attr("timeout: int")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
    .Doc(R"doc(
A Reader that outputs the lines of a file delimited by '\n'.

reader_handle: The handle to reference the Reader.
container: If non-empty, this reader is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this reader is named in the given bucket
  with this shared_name. Otherwise, the node name is used instead.
servers: The list of bootstrap servers, separated by ','.
group: The consumer group id.
eof: If True, the kafka reader will stop on EOF.
timeout: The timeout value for the Kafka Consumer to wait (in millisecond).
)doc");

}  // namespace tensorflow
