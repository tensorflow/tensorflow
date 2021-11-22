/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using tensorflow::tpu::TPUEmbeddingConfiguration;

REGISTER_OP("_ExecuteTPUEmbeddingPartitioner")
    .Output("common_config: string")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> Status {
      string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      TPUEmbeddingConfiguration config;
      TF_RET_CHECK(config.ParseFromString(config_string));
      if (config.mode() == TPUEmbeddingConfiguration::UNSPECIFIED) {
        return errors::InvalidArgument(
            "TPUEmbeddingConfiguration.mode is INVALID.  Must be INFERENCE, "
            "TRAINING, or BACKWARD_PASS_ONLY");
      }
      c->set_output(0, c->Scalar());
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

An op that executes the TPUEmbedding partitioner on the central configuration
device and computes the HBM size (in bytes) required for TPUEmbedding operation.

common_config: A string-encoded tpu_embedding::CommonConfiguration proto
containing metadata about the TPUEmbedding partitioner output and
the HBM size (in bytes) required for operation.
config: An TPUEmbeddingConfiguration proto serialized to a string,
describing the desired TPUEmbedding configuration.
)doc");

REGISTER_OP("_ConfigureTPUEmbeddingMemory")
    .Input("common_config: string")
    .Output("task_host_config: string")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> Status {
      string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      TPUEmbeddingConfiguration config;
      TF_RET_CHECK(config.ParseFromString(config_string));
      if (config.mode() == TPUEmbeddingConfiguration::UNSPECIFIED) {
        return errors::InvalidArgument(
            "TPUEmbeddingConfiguration.mode is INVALID.  Must be INFERENCE, "
            "TRAINING, or BACKWARD_PASS_ONLY");
      }
      TF_RET_CHECK(c->num_inputs() == 1);
      // Validate that all the input shape is compatible.
      ShapeHandle input(c->Scalar());
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), input, &input));
      c->set_output(0, c->Scalar());
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

An op that configures the TPUEmbedding software on a host.

common_config: A string-encoded tpu_embedding CommonConfiguration proto
containing metadata about the TPUEmbedding partitioner output and the HBM
size (in bytes) required for operation.
task_host_config: A string-encoded tpu_embedding PerHostConfiguration proto
containing metadata about the memory allocations reserved for TPUEmbedding.
config: An TPUEmbeddingConfiguration proto serialized to a string,
describing the desired TPUEmbedding configuration.
)doc");

REGISTER_OP("_ConfigureTPUEmbeddingHost")
    .Input("common_config: string")
    .Input("task_host_config: N * string")
    .Output("host_config: string")
    .Attr("N: int >= 1")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> Status {
      string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      TPUEmbeddingConfiguration config;
      TF_RET_CHECK(config.ParseFromString(config_string));
      if (config.mode() == TPUEmbeddingConfiguration::UNSPECIFIED) {
        return errors::InvalidArgument(
            "TPUEmbeddingConfiguration.mode is INVALID.  Must be INFERENCE, "
            "TRAINING, or BACKWARD_PASS_ONLY");
      }
      TF_RET_CHECK(c->num_inputs() > 0);
      ShapeHandle input(c->Scalar());
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), input, &input));
      c->set_output(0, c->Scalar());
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

An op that configures the TPUEmbedding software on a host.

common_config: A string-encoded tpu_embedding CommonConfiguration proto
containing metadata about the TPUEmbedding partitioner output.
task_host_config: A string-encoded tpu_embedding PerHostConfiguration proto from
each host containing metadata about the memory allocations reserved for
TPUEmbedding.
config: An TPUEmbeddingConfiguration proto serialized to a string,
describing the desired TPUEmbedding configuration.
)doc");

REGISTER_OP("_ConnectInterTPUEmbeddingCommunication")
    .Input("host_config: N * string")
    .Attr("N: int >= 1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> Status {
      TF_RET_CHECK(c->num_inputs() > 0);
      ShapeHandle input(c->Scalar());
      // Validate that all the inputs are compatible with the correct
      // vector shape.
      for (int i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->Merge(c->input(i), input, &input));
      }
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

An op that sets up communication between TPUEmbedding host software instances after
ConfigureTPUEmbeddingHost has been called on each host.

host_config: A string-encoded tpu_embedding PerHostConfiguration proto read
from each host containing metadata about the RPC port used for communication
with that host.
)doc");

REGISTER_OP("_FinalizeTPUEmbeddingSystemConfiguration")
    .Input("host_config: N * string")
    .Attr("N: int >= 1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> Status {
      ShapeHandle input(c->Scalar());
      // Validate that all the inputs are compatible with the correct
      // vector shape.
      for (int i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->Merge(c->input(i), input, &input));
      }
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

An op that finalizes the TPUEmbedding system configuration after
ConfigureTPUEmbeddingHost has been called on each host.

host_config: A string-encoded tpu_embedding PerHostConfiguration proto read
from each host containing metadata about the HBM base byte address reserved for
the TPUEmbedding on that host.
)doc");

}  // namespace tensorflow
