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
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/kernels/hexagon/hexagon_control_wrapper.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class RemoteFusedGraphExecuteOp : public OpKernel {
 public:
  explicit RemoteFusedGraphExecuteOp(OpKernelConstruction* const ctx)
      : OpKernel(ctx), graph_transferer_() {
    string serialized_proto;
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("serialized_graph_transfer_info", &serialized_proto));
    graph_transferer_.SetSerializedGraphTransferInfo(serialized_proto);
    const GraphTransferInfo& gt_info = graph_transferer_.GetGraphTransferInfo();
    switch (gt_info.destination()) {
      case GraphTransferInfo::NOP:
        break;
      case GraphTransferInfo::HEXAGON:
        soc_control_wrapper_.reset(new HexagonControlWrapper());
        break;
      default:
        // Other destination is not supported yet.
        CHECK(false);
        break;
    }

    if (soc_control_wrapper_) {
      // 1. Initialize remote processor
      soc_control_wrapper_->Init();

      // 2. Setup graph in remote processor
      soc_control_wrapper_->SetupGraph(graph_transferer_);
    }
  }

  ~RemoteFusedGraphExecuteOp() final {
    if (soc_control_wrapper_) {
      // 6. Teardown graph in remote processor
      soc_control_wrapper_->TeardownGraph();

      // 7. Finalize remote processor
      soc_control_wrapper_->Finalize();
    }
  }

  void Compute(OpKernelContext* const ctx) final {
    CHECK(ctx != nullptr);
    const int input_count = ctx->num_inputs();
    const GraphTransferInfo& gt_info = graph_transferer_.GetGraphTransferInfo();
    CHECK(input_count == gt_info.graph_input_node_info_size())
        << "input_count = " << input_count
        << ", gt input count = " << gt_info.graph_input_node_info_size();

    // 3. Send inputs into remote processor
    for (int i = 0; i < input_count; ++i) {
      const Tensor& input_tensor = ctx->input(i);
      const GraphTransferInfo::GraphInputNodeInfo& input_node_info =
          gt_info.graph_input_node_info(i);
      const string& input_node_name = input_node_info.name();
      if (soc_control_wrapper_) {
        soc_control_wrapper_->FillInputNode(input_node_name, input_tensor);
      }
    }

    // 4. Execute graph in remote processor
    if (soc_control_wrapper_) {
      soc_control_wrapper_->ExecuteGraph();
    }

    // 5. Load outputs from remote processor
    const int output_count = ctx->num_outputs();
    CHECK(output_count == gt_info.graph_output_node_info_size());
    for (int i = 0; i < output_count; ++i) {
      Tensor* output = nullptr;
      TensorShape output_shape;
      const GraphTransferInfo::GraphOutputNodeInfo& output_node_info =
          gt_info.graph_output_node_info(i);
      for (const int64 dim : output_node_info.shape()) {
        output_shape.AddDim(dim);
      }
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &output));
      if (soc_control_wrapper_) {
        std::vector<ISocControlWrapper::ByteArray> outputs;
        soc_control_wrapper_->ReadOutputNode(output_node_info.name(), &outputs);
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
  GraphTransferer graph_transferer_;
  std::unique_ptr<ISocControlWrapper> soc_control_wrapper_;

  TF_DISALLOW_COPY_AND_ASSIGN(RemoteFusedGraphExecuteOp);
};

REGISTER_KERNEL_BUILDER(Name("RemoteFusedGraphExecute").Device(DEVICE_CPU),
                        RemoteFusedGraphExecuteOp);

}  // namespace tensorflow
