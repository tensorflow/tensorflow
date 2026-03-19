
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Produced by generate_tpu_embedding_load_retrieve_ops.py (Google-internal).

#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/tpu/optimization_parameters.pb.h"
#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"

namespace tensorflow {
namespace tpu {

REGISTER_OP("LoadTPUEmbeddingAdagradParameters")
    .Input("parameters: float32")
    .Input("accumulators: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingAdagradParameters")
    .Output("parameters: float32")
    .Output("accumulators: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingAdagradMomentumParameters")
    .Input("parameters: float32")
    .Input("accumulators: float32")
    .Input("momenta: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingAdagradMomentumParameters")
    .Output("parameters: float32")
    .Output("accumulators: float32")
    .Output("momenta: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingStochasticGradientDescentParameters")
    .Input("parameters: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingStochasticGradientDescentParameters")
    .Output("parameters: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingFTRLParameters")
    .Input("parameters: float32")
    .Input("accumulators: float32")
    .Input("linears: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingFTRLParameters")
    .Output("parameters: float32")
    .Output("accumulators: float32")
    .Output("linears: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingADAMParameters")
    .Input("parameters: float32")
    .Input("momenta: float32")
    .Input("velocities: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingADAMParameters")
    .Output("parameters: float32")
    .Output("momenta: float32")
    .Output("velocities: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingMomentumParameters")
    .Input("parameters: float32")
    .Input("momenta: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingMomentumParameters")
    .Output("parameters: float32")
    .Output("momenta: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingRMSPropParameters")
    .Input("parameters: float32")
    .Input("ms: float32")
    .Input("mom: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingRMSPropParameters")
    .Output("parameters: float32")
    .Output("ms: float32")
    .Output("mom: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingCenteredRMSPropParameters")
    .Input("parameters: float32")
    .Input("ms: float32")
    .Input("mom: float32")
    .Input("mg: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingCenteredRMSPropParameters")
    .Output("parameters: float32")
    .Output("ms: float32")
    .Output("mom: float32")
    .Output("mg: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingMDLAdagradLightParameters")
    .Input("parameters: float32")
    .Input("accumulators: float32")
    .Input("weights: float32")
    .Input("benefits: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingMDLAdagradLightParameters")
    .Output("parameters: float32")
    .Output("accumulators: float32")
    .Output("weights: float32")
    .Output("benefits: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingAdadeltaParameters")
    .Input("parameters: float32")
    .Input("accumulators: float32")
    .Input("updates: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingAdadeltaParameters")
    .Output("parameters: float32")
    .Output("accumulators: float32")
    .Output("updates: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingProximalAdagradParameters")
    .Input("parameters: float32")
    .Input("accumulators: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingProximalAdagradParameters")
    .Output("parameters: float32")
    .Output("accumulators: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingProximalYogiParameters")
    .Input("parameters: float32")
    .Input("v: float32")
    .Input("m: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingProximalYogiParameters")
    .Output("parameters: float32")
    .Output("v: float32")
    .Output("m: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

REGISTER_OP("LoadTPUEmbeddingFrequencyEstimatorParameters")
    .Input("parameters: float32")
    .Input("last_hit_step: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(LoadOpShapeFunction());

REGISTER_OP("RetrieveTPUEmbeddingFrequencyEstimatorParameters")
    .Output("parameters: float32")
    .Output("last_hit_step: float32")
    .Attr("table_id: int = -1")
    .Attr("table_name: string = \"\"")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .Attr("config: string = \"\"")
    .SetIsStateful()
    .SetShapeFn(RetrieveOpShapeFunction());

}  // namespace tpu
}  // namespace tensorflow
