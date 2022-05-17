/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// Ops to load and retrieve embeddings for TPU Embedding.

#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_EMBEDDING_LOAD_RETRIEVE_OPS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_EMBEDDING_LOAD_RETRIEVE_OPS_H_

#include <array>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"

namespace tensorflow {

// Computes (and VLOGs) the expected shapes of the embedding table shards.
Status ComputeExpectedTableShardShapes(
    const tpu::TPUEmbeddingConfiguration& config, int shard_id, int num_shards,
    const string& op_name, std::vector<TensorShape>* table_shapes);

// The LoadAllTPUEmbeddingParameters op is used to load initial embedding
// table parameters onto a host that has already been configured using
// ConfigureTPUEmbeddingHost. This Op should be used when TPUEmbedding is part
// of a training loop. The Op takes four input lists of tensors. Each list has
// one entry per embedding table, but some entries are ignored based on the
// particular optimization algorithm used for each table. parameters is the
// initial values of the embedding tables, and auxiliary[1-3] are the initial
// values of the auxiliary parameters.
class LoadAllTPUEmbeddingParametersOp : public OpKernel {
 public:
  explicit LoadAllTPUEmbeddingParametersOp(OpKernelConstruction* ctx);
  ~LoadAllTPUEmbeddingParametersOp() override = default;

  void Compute(OpKernelContext* ctx) override;

 protected:
  // Checks if shard_id is a tensor input of `ctx` and if it is, validates and
  // sets the value to `shard_id`.
  static Status ValidateAndSetShardId(OpKernelContext* ctx, int* shard_id);

  void GetStateVariables(
      OpKernelContext* ctx,
      std::array<std::vector<absl::Span<const float>>,
                 tpu::kMaxAuxiliaryParameterCount + 1>& state_variable_vector);

  // Identifying shard of this operation.
  int shard_id_ = -1;

  // Number of shards the embedding tables are divided into.
  int num_shards_ = -1;

  // Flag identifying whether the table_shapes was initialized.
  bool table_shape_initialized_ = false;
  tpu::TPUEmbeddingConfiguration config_;
  std::vector<TensorShape> table_shapes_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LoadAllTPUEmbeddingParametersOp);
};

// The RetrieveAllTPUEmbeddingParameters op is used to retrieve updated
// embedding table parameters from a TPU that has already been
// configured using ConfigureTPUEmbeddingHostOp. This Op should be used when
// TPUEmbedding is part of a training loop. The Op returns four output lists of
// tensors. Each list has one entry per embedding table, but entries are empty
// when the relevant table does not have that number of auxiliary parameters.
// The parameters output is the updated values of the embedding tables, and
// auxiliary[1-3] are the updated values of the auxiliary parameters.

// Currently, this op is the only method to make sure that the TPUEmbedding has
// completed execution of the mini-batches enqueued so far.
// TODO(misard, b/34936670): Add a TensorFlow op that waits till all
// minibatches have been processed by the TPUEmbedding on the current host.
class RetrieveAllTPUEmbeddingParametersOp : public OpKernel {
 public:
  explicit RetrieveAllTPUEmbeddingParametersOp(OpKernelConstruction* ctx);
  ~RetrieveAllTPUEmbeddingParametersOp() override = default;

  void Compute(OpKernelContext* ctx) override;

 protected:
  void GetStateVariables(
      OpKernelContext* ctx,
      std::array<std::vector<absl::Span<float>>,
                 tpu::kMaxAuxiliaryParameterCount + 1>& state_variable_vector,
      std::vector<int>& num_state_variables);

  tpu::TPUEmbeddingConfiguration config_;
  std::vector<TensorShape> table_shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(RetrieveAllTPUEmbeddingParametersOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_EMBEDDING_LOAD_RETRIEVE_OPS_H_
