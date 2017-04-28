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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class RemoteFusedGraphExecuteOp : public OpKernel {
 public:
  explicit RemoteFusedGraphExecuteOp(OpKernelConstruction* const ctx)
      : OpKernel(ctx), execute_info_() {
    string serialized_proto;
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("serialized_remote_fused_graph_execute_info",
                                &serialized_proto));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tinputs", &input_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutputs", &output_types_));
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
    const int graph_input_count = execute_info_.graph_input_node_name_size();
    CHECK(input_count == graph_input_count &&
          input_count == input_types_.size())
        << "input_count = " << input_count
        << ", gt input count = " << execute_info_.graph_input_node_name_size()
        << ", type count = " << input_types_.size();

    // 3. Send first data type inputs into remote processor
    for (int i = 0; i < graph_input_count; ++i) {
      const Tensor& input_tensor = ctx->input(i);
      const string& input_node_name = execute_info_.graph_input_node_name(i);
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
    CHECK(output_count == execute_info_.graph_output_node_name_size() &&
          output_count == output_types_.size());
    for (int i = 0; i < output_count; ++i) {
      Tensor* output = nullptr;
      const string& output_node_name = execute_info_.graph_output_node_name(i);
      if (remote_fused_graph_executor_) {
        remote_fused_graph_executor_->ReadOutputNode(
            output_node_name,
            [i, &ctx, &output](const TensorShape& shape) -> Tensor* {
              TF_CHECK_OK(ctx->allocate_output(i, shape, &output));
              return output;
            });
      }
    }
  }

  bool IsExpensive() final { return true; }

 private:
  RemoteFusedGraphExecuteInfo execute_info_;
  std::unique_ptr<IRemoteFusedGraphExecutor> remote_fused_graph_executor_;
  DataTypeVector input_types_;
  DataTypeVector output_types_;

  TF_DISALLOW_COPY_AND_ASSIGN(RemoteFusedGraphExecuteOp);
};

REGISTER_KERNEL_BUILDER(Name("RemoteFusedGraphExecute").Device(DEVICE_CPU),
                        RemoteFusedGraphExecuteOp);

}  // namespace tensorflow
