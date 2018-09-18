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

REGISTER_OP("IgniteDataset")
    .Input("cache_name: string")
    .Input("host: string")
    .Input("port: int32")
    .Input("local: bool")
    .Input("part: int32")
    .Input("page_size: int32")
    .Input("schema: int32")
    .Input("permutation: int32")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
IgniteDataset that allows to get data from Apache Ignite.

Apache Ignite is a memory-centric distributed database, caching, and processing
platform for transactional, analytical, and streaming workloads, delivering 
in-memory speeds at petabyte scale. This contrib package contains an 
integration between Apache Ignite and TensorFlow. The integration is based on 
tf.data from TensorFlow side and Binary Client Protocol from Apache Ignite side. 
It allows to use Apache Ignite as a datasource for neural network training, 
inference and all other computations supported by TensorFlow. Ignite Dataset
is based on Apache Ignite Binary Client Protocol.

cache_name: Ignite Cache Name.
host: Ignite Thin Client Host.
port: Ignite Thin Client Port.
local: Local flag that defines that data should be fetched from local host only.
part: Partition data should be fetched from.
page_size: Page size for Ignite Thin Client.
schema: Internal structure that defines schema of cache objects.
permutation: Internal structure that defines permutation of cache objects.
)doc");

}  // namespace tensorflow
