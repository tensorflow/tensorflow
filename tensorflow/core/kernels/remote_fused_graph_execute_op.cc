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

// See docs in ../ops/remote_fused_graph_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/remote_fused_graph_execute_info.pb.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_executor.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class RemoteFusedGraphExecuteOp : public OpKernel {
 public:
  explicit RemoteFusedGraphExecuteOp(OpKernelConstruction* const ctx)
      : OpKernel(ctx), execute_info_() {
    string serialized_proto;
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("serialized_graph_transfer_info", &serialized_proto));
    execute_info_.ParseFromString(serialized_proto);
    if (!execute_info_.executor_name().empty()) {
      const RemoteFusedGraphExecuteUtils::ExecutorBuildFunc* build_func =
          RemoteFusedGraphExecuteUtils::GetExecutorBuildFunc(
              execute_info_.executor_name());
      if (build_func != nullptr) {
        Status status = (*build_func)(&remote_fused_graph_executor_);
      } else {
        LOG(ERROR) << "Executor not found for "
                   << execute_info_.executor_name();
      }
    }

    if (remote_fused_graph_executor_) {
      // 1. Initialize remote processor
      remote_fused_graph_executor_->Init(execute_info_);
      // Explicitly clear serialized executor parameter after initialization
      // to release unnecessary memory.
      execute_info_.clear_serialized_executor_parameters();

      // 2. Setup graph in remote processor
      remote_fused_graph_executor_->SetupGraph();
    }
  }

  ~RemoteFusedGraphExecuteOp() final {
    if (remote_fused_graph_executor_) {
      // 6. Teardown graph in remote processor
      remote_fused_graph_executor_->TeardownGraph();

      // 7. Finalize remote processor
      remote_fused_graph_executor_->Finalize();
    }
  }

  void Compute(OpKernelContext* const ctx) final {
    CHECK(ctx != nullptr);
    const int input_count = ctx->num_inputs();
    CHECK(input_count == execute_info_.graph_input_node_info_size())
        << "input_count = " << input_count
        << ", gt input count = " << execute_info_.graph_input_node_info_size();

    // 3. Send inputs into remote processor
    for (int i = 0; i < input_count; ++i) {
      const Tensor& input_tensor = ctx->input(i);
      const RemoteFusedGraphExecuteInfo::GraphIONodeInfo& input_node_info =
          execute_info_.graph_input_node_info(i);
      const string& input_node_name = input_node_info.name();
      if (remote_fused_graph_executor_) {
        remote_fused_graph_executor_->FillInputNode(input_node_name,
                                                    input_tensor);
      }
    }

    // 4. Execute graph in remote processor
    if (remote_fused_graph_executor_) {
      remote_fused_graph_executor_->ExecuteGraph();
    }

    // 5. Load outputs from remote processor
    const int output_count = ctx->num_outputs();
    CHECK(output_count == execute_info_.graph_output_node_info_size());
    for (int i = 0; i < output_count; ++i) {
      Tensor* output = nullptr;
      TensorShape output_shape;
      const RemoteFusedGraphExecuteInfo::GraphIONodeInfo& output_node_info =
          execute_info_.graph_output_node_info(i);
      for (const int64 dim : output_node_info.shape()) {
        output_shape.AddDim(dim);
      }
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &output));
      if (remote_fused_graph_executor_) {
        std::vector<IRemoteFusedGraphExecutor::ByteArray> outputs;
        remote_fused_graph_executor_->ReadOutputNode(output_node_info.name(),
                                                     &outputs);
        // TODO(satok): Remove this check (<= 1). And support multiple outputs
        // for each output node
        CHECK(outputs.size() <= 1);
        if (!outputs.empty()) {
          CHECK(output->TotalBytes() >= std::get<1>(outputs[0]));
          // TODO(satok): Avoid specifying float
          std::memcpy(output->flat<float>().data(), std::get<0>(outputs[0]),
                      std::get<1>(outputs[0]));
        }
      }
    }
  }

  bool IsExpensive() final { return true; }

 private:
  RemoteFusedGraphExecuteInfo execute_info_;
  std::unique_ptr<IRemoteFusedGraphExecutor> remote_fused_graph_executor_;

  TF_DISALLOW_COPY_AND_ASSIGN(RemoteFusedGraphExecuteOp);
};

REGISTER_KERNEL_BUILDER(Name("RemoteFusedGraphExecute").Device(DEVICE_CPU),
                        RemoteFusedGraphExecuteOp);

}  // namespace tensorflow
