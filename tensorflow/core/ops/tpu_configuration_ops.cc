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
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// Configuring a distributed TPU system is achieved by running
// the following Ops:
//
// 1 Run _DisconnectHostFromDistributedTPUSystem on the TPU_SYSTEM of each
// host. This is needed in case the system had previously been configured. It
// returns, for each host, the number of TPU chips on the host.
//
// 2 Run _ConfigureDistributedTPU on TPU_SYSTEM of worker 0. Takes as input the
// number of chips on each host. Validates that all hosts have the same number
// of chips, and that the chips are consistent with the topology set by
// flags. Has a single output which is a proto describing the requested system
// configuration, which is sent to all hosts.
//
// 3 Run _InitializeHostForDistributedTPU on the TPU_SYSTEM of each host, taking
// as input the output from ConfigureDistributedTPU. Has a single Tensor output
// which is a vector of int32 indicating, for each TPU on the host, what its
// global TPU system id is.
//
// 4 Run _WaitForDistributedTPU on TPU_SYSTEM, taking as input the
// outputs from all the _InitializeHostForDistributedTPU
// Ops. _These partial specs are combined in the Op with the outputs from
// the host initialization Ops to construct a mapping from full TPU device
// specs to global TPU ids. Has a single Tensor output which is a
// matrix of int32 indicating, for each host (outer dimension) and for
// each TPU on the host (inner dimension) what that TPU's global id
// is. _WaitForDistributedTPU also waits for the TPU distributed
// system to initialize fully, which may take several minutes for a
// large system.
//
// 5 Run _SetGlobalTPUArray on the TPU_SYSTEM of each host, taking as input the
// output from _WaitForDistributedTPU. This Op tells each host the global Id of
// every TPU on every host.
//
// Most user code works by placing the ConfigureDistributedTPU Op on the desired
// TPU_SYSTEM device, and a graph rewrite replaces it by the subgraph described
// above.
//
//
// A distributed TPU system can be cleanly shut down by running the following
// Ops:
//
// 1 Run _DisconnectHostFromDistributedTPUSystem on the TPU_SYSTEM of each host.
//
// 2 Run _ShutdownDistributedTPU on the TPU_SYSTEM where
// _ConfigureDistributedTPU was run. The Op will return an error if no system is
// configured.
//
//
// Most user code works by placing the ShutdownDistributedTPU Op on the desired
// TPU_SYSTEM device, and a graph rewrite replaces it by the subgraph described
// above.

REGISTER_OP("_ConfigureDistributedTPU")
    .Input("inputs: N * int32")
    .Output("output: string")
    .Attr("N: int >= 1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      // Validate that all the inputs are scalars.
      for (int i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &input));
      }
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
An op that sets up the centralized structures for a distributed TPU
system.

inputs: A scalar tensor for each host indicating how many TPU chips
there are on the host.
output: A tensor containing a TPUHostConfiguration proto serialized to
a string, containing the information necessary to initialize the chips
in a host.
)doc");

REGISTER_OP("_WaitForDistributedTPU")
    .Input("inputs: N * int32")
    .Output("topology: string")
    .Attr("startup_timeout_sec: int = 20")
    .Attr("N: int")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      // Validate that all the inputs have the same vector shape.
      for (int i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &input));
      }
      c->set_output(0, c->Scalar());
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
An op that blocks execution until a distributed TPU system has
started up. This Op must be run on the same TPU_SYSTEM device as
_ConfigureDistributedTPU, and takes an inputs the outputs from the
_InitializeHostForDistributedTPU Ops.

inputs: For each initialized host, a vector giving the global TPU id
of each TPU on the host.
topology: A serialized tensorflow.tpu.TopologyProto that describes the TPU
topology.
startup_timeout_sec: The number of seconds to wait for the TPU system
to stabilize.
)doc");

REGISTER_OP("_SetGlobalTPUArray")
    .Input("topology: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
An op that informs a host of the global ids of all the of TPUs in the
system.

topology: A serialized tensorflow.tpu.TopologyProto that describes the TPU
topology.
)doc");

REGISTER_OP("_ShutdownDistributedTPU")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
An op that shuts down a running distributed TPU system. The Op returns
an error if no system is running. This Op must be run on the same
TPU_SYSTEM device as the corresponding _ConfigureDistributedTPU was run
to start the system, and must be run only after
_DisconnectHostFromDistributedTPUSystem has completed on every host in
the system.
)doc");

REGISTER_OP("_InitializeHostForDistributedTPU")
    .Input("input: string")
    .Output("tpu_ids: int32")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input));
      c->set_output(0, c->Vector(c->UnknownDim()));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
An op that connects each chip on the host to a centralized UberDriver to allow
them to operate as a distributed system with chips in other hosts.

input: A string containing the address of the UberDriver to connect to.
tpu_ids: A vector containing the global TPU id of each TPU on the host.
)doc");

REGISTER_OP("_DisconnectHostFromDistributedTPUSystem")
    .Output("number_of_tpu_chips: int32")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
An op that disconnects the TPUs on a host from a running distributed
TPU system.

number_of_tpu_chips: A scalar tensor containing the number of TPU
chips on the host.
)doc");

REGISTER_OP("ConfigureDistributedTPU")
    .Output("topology: string")
    .Attr("embedding_config: string = ''")
    .Attr("tpu_embedding_config: string = ''")
    .Attr("is_global_init: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ShutdownDistributedTPU")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}  // end namespace tensorflow
