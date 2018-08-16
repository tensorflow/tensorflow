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
  .Input("distributed: bool")
  .Input("page_size: int32")
  .Input("username: string")
  .Input("password: string")
  .Input("certfile: string")
  .Input("keyfile: string")
  .Input("cert_password: string")
  .Input("schema: int32")
  .Input("permutation: int32")
  .Output("handle: variant")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape)
  .Doc(R"doc(
Ignite Dataset...

cache_name: Cache Name.
host: Host.
port: Port.
local: Local.
part: Part.
distributed: Distributed.
page_size: Page size.
username: Username.
password: Password.
certfile: SSL certificate.
keyfile: Private key file.
cert_password: SSL certificate password.
schema: Schema.
permutation: Permutation.
)doc");

}  // namespace tensorflow
