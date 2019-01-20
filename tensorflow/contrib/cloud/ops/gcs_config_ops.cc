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

REGISTER_OP("GcsConfigureCredentials")
    .Input("json: string")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Configures the credentials used by the GCS client of the local TF runtime.

The json input can be of the format:

1. Refresh Token:
{
  "client_id": "<redacted>",
  "client_secret": "<redacted>",
  "refresh_token: "<redacted>",
  "type": "authorized_user",
}

2. Service Account:
{
  "type": "service_account",
  "project_id": "<redacted>",
  "private_key_id": "<redacted>",
  "private_key": "------BEGIN PRIVATE KEY-----\n<REDACTED>\n-----END PRIVATE KEY------\n",
  "client_email": "<REDACTED>@<REDACTED>.iam.gserviceaccount.com",
  "client_id": "<REDACTED>",
  # Some additional fields elided
}

Note the credentials established through this method are shared across all
sessions run on this runtime.

Note be sure to feed the inputs to this op to ensure the credentials are not
stored in a constant op within the graph that might accidentally be checkpointed
or in other ways be persisted or exfiltrated.
)doc");

REGISTER_OP("GcsConfigureBlockCache")
    .Input("max_cache_size: uint64")
    .Input("block_size: uint64")
    .Input("max_staleness: uint64")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Re-configures the GCS block cache with the new configuration values.

If the values are the same as already configured values, this op is a no-op. If
they are different, the current contents of the block cache is dropped, and a
new block cache is created fresh.
)doc");

}  // namespace tensorflow
