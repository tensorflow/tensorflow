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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/sharding_builder.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/lower_function_call_op.h"
#include "tensorflow/core/common_runtime/lower_if_op.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace tensorflow {

namespace {

// TODO(phawkins) add a canonical copy of these operator names and refactor
// everything to use it.
static const char* const kSendFromHostOp = "_XlaSendFromHost";
static const char* const kRecvAtHostOp = "_XlaRecvAtHost";

absl::Status MakeXlaShapes(absl::Span<const TensorShape> shapes,
                           absl::Span<const DataType> dtypes,
                           std::vector<xla::Shape>* xla_shapes,
                           xla::Shape* xla_shape) {
  for (int i = 0; i < shapes.size(); i++) {
    xla::Shape single_xla_shape;
    TF_RETURN_IF_ERROR(
        TensorShapeToXLAShape(dtypes[i], shapes[i], &single_xla_shape));
    VLOG(2) << "Shape " << single_xla_shape.DebugString();
    xla_shapes->push_back(single_xla_shape);
  }
  // Temporarily add a dummy output to the shape array before making the tuple:
  // this output is used for control dependencies between host compute ops.
  xla_shapes->push_back(xla::ShapeUtil::MakeShape(xla::PRED, {}));
  *xla_shape = xla::ShapeUtil::MakeTupleShape(*xla_shapes);
  // Remove the dummy output from the vector that will be used to copy real
  // outputs from host to device.
  xla_shapes->pop_back();
  return absl::OkStatus();
}

// This TensorFlow pseudo-op is used to record host-side computation.
class HostComputeOp : public XlaOpKernel {
 public:
  explicit HostComputeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cost_estimate_ns", &cost_estimate_));

    std::string key;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_key", &send_key_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_key", &recv_key_));

    // If any of the send or recv keys is set to the default value, use the
    // `key` attribute for it. Old bridge uses the same key for both send and
    // recv unlike the MLIR bridge that uses different keys.
    if (send_key_.empty()) send_key_ = key;
    if (recv_key_.empty()) recv_key_ = key;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("tpu_core", &tpu_core_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tinputs", &input_dtypes_));
    OP_REQUIRES(ctx, ctx->num_inputs() == input_dtypes_.size(),
                errors::InvalidArgument("Tinputs size=", input_dtypes_.size(),
                                        " but expected ", ctx->num_inputs(),
                                        " inputs."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutputs", &output_dtypes_));
    OP_REQUIRES(ctx, ctx->num_outputs() == output_dtypes_.size(),
                errors::InvalidArgument("Toutputs size=", output_dtypes_.size(),
                                        " but expected ", ctx->num_outputs(),
                                        " outputs."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ancestors", &ancestors_));
    NameAttrList shape_inference_graph;
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("shape_inference_graph", &shape_inference_graph));
    const std::string& shape_inference_func_name = shape_inference_graph.name();
    if (shape_inference_func_name.empty()) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &static_output_shapes_));
      OP_REQUIRES(ctx, static_output_shapes_.size() == output_dtypes_.size(),
                  errors::InvalidArgument(
                      "shapes attr list size ", static_output_shapes_.size(),
                      " differs from dtypes size ", output_dtypes_.size()));
      OP_REQUIRES_OK(ctx, MakeXlaShapes(static_output_shapes_, output_dtypes_,
                                        &static_xla_output_shapes_,
                                        &static_xla_output_shape_));
      VLOG(2) << "Output Shape: " << static_xla_output_shape_.DebugString();
    } else {
      FunctionLibraryRuntime* flib_runtime = ctx->function_library();
      OP_REQUIRES(ctx, flib_runtime != nullptr,
                  errors::Internal(
                      "No function library runtime at kernel construction"));
      const FunctionLibraryDefinition* library =
          flib_runtime->GetFunctionLibraryDefinition();
      const FunctionDef* fdef = library->Find(shape_inference_func_name);
      OP_REQUIRES(
          ctx, fdef != nullptr,
          errors::Internal("Failed to find function ",
                           shape_inference_func_name, " in function library."));
      OP_REQUIRES_OK(ctx, FunctionDefToBodyHelper(
                              *fdef, AttrSlice(&shape_inference_graph.attr()),
                              library, &shape_inference_graph_function_));
      VLOG(2) << "Output Shape to be inferred at compile time";
    }
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kXlaTokenInputNodesAttrName, &token_input_nodes_));
    OP_REQUIRES(ctx, !token_input_nodes_.empty(),
                errors::InvalidArgument("XlaHostCompute node does not have ",
                                        kXlaTokenInputNodesAttrName, " attr"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kXlaOriginalOutsideCompilationNodeName,
                                     &original_node_name_));
  }

  ~HostComputeOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    XlaCompiler* compiler = ctx->compiler();

    std::vector<xla::XlaOp> input_handles;
    std::vector<TensorShape> input_shapes;
    auto inputs = ctx->InputList("inputs", &input_handles, &input_shapes);
    const auto device_sharding = xla::sharding_builder::AssignDevice(tpu_core_);
    xla::XlaScopedShardingAssignment assign_sharding(b, device_sharding);

    std::vector<xla::XlaOp> input_tokens;
    for (auto& token_input_node : token_input_nodes_) {
      auto token_or = compiler->GetNodeToken(token_input_node);
      OP_REQUIRES_OK(ctx, token_or.status());
      input_tokens.push_back(token_or.value());
    }
    xla::XlaOp token = xla::AfterAll(b, input_tokens);

    // Send values to the host.
    std::vector<xla::XlaOp> send_to_host_tokens;
    for (int i = 0; i < input_handles.size(); ++i) {
      const string channel_name = GetDeviceToHostChannelName(send_key_, i);
      xla::Shape xla_shape;
      OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(input_dtypes_[i],
                                                input_shapes[i], &xla_shape));
      // Specify frontend attributes.
      xla::FrontendAttributes attrs;
      (*attrs.mutable_map())[xla::kXlaHostTransferRendezvousNameAttr] =
          channel_name;
      (*attrs.mutable_map())[xla::kXlaHostTransferHandlerNameAttr] =
          xla::kXlaHostTransferTfRendezvousHandlerName;
      b->SetFrontendAttributes(attrs);
      xla::ChannelHandle channel;
      OP_REQUIRES_OK(
          ctx, compiler->GetDeviceToHostChannelHandle(channel_name, &channel));
      send_to_host_tokens.push_back(
          xla::SendToHost(input_handles[i], token, xla_shape, channel));
      b->ClearOpMetadata();
    }
    xla::XlaOp recv_from_host_token_input =
        send_to_host_tokens.empty() ? token
                                    : xla::AfterAll(b, send_to_host_tokens);
    if (!input_handles.empty()) {
      // Register the shapes used in this transfer.
      OP_REQUIRES_OK(ctx, ctx->compiler()->SetDeviceToHostMetadata(
                              send_key_, input_dtypes_, input_shapes));
    }
    // Compute the shapes of the values to copy to the device, if necessary.
    std::vector<TensorShape>* output_shapes;
    std::vector<xla::Shape>* xla_output_shapes;
    xla::Shape* xla_output_shape;
    std::vector<TensorShape> inferred_output_shapes;
    std::vector<xla::Shape> inferred_xla_output_shapes;
    xla::Shape inferred_xla_output_shape;
    if (shape_inference_graph_function_) {
      OP_REQUIRES_OK(
          ctx, InferOutputShapes(
                   ctx, ctx->function_library()->GetFunctionLibraryDefinition(),
                   &inferred_output_shapes));
      OP_REQUIRES_OK(ctx, MakeXlaShapes(inferred_output_shapes, output_dtypes_,
                                        &inferred_xla_output_shapes,
                                        &inferred_xla_output_shape));
      output_shapes = &inferred_output_shapes;
      xla_output_shapes = &inferred_xla_output_shapes;
      xla_output_shape = &inferred_xla_output_shape;
    } else {
      output_shapes = &static_output_shapes_;
      xla_output_shapes = &static_xla_output_shapes_;
      xla_output_shape = &static_xla_output_shape_;
    }
    OP_REQUIRES(
        ctx, output_shapes->size() == ctx->num_outputs(),
        errors::InvalidArgument("Op has ", ctx->num_outputs(), " outputs ",
                                " but output shape vector of size ",
                                output_shapes->size()));
    if (ctx->num_outputs() > 0) {
      // Register the shapes used in this transfer.
      OP_REQUIRES_OK(ctx, ctx->compiler()->SetHostToDeviceMetadata(
                              recv_key_, output_dtypes_, *output_shapes));
    }
    // Copy results to the device.
    std::vector<xla::XlaOp> recv_from_host_tokens;
    for (int i = 0; i < output_shapes->size(); ++i) {
      const string channel_name = GetHostToDeviceChannelName(recv_key_, i);
      // Specify frontend attributes.
      xla::FrontendAttributes attrs;
      (*attrs.mutable_map())[xla::kXlaHostTransferRendezvousNameAttr] =
          channel_name;
      (*attrs.mutable_map())[xla::kXlaHostTransferHandlerNameAttr] =
          xla::kXlaHostTransferTfRendezvousHandlerName;
      b->SetFrontendAttributes(attrs);
      xla::ChannelHandle channel;
      OP_REQUIRES_OK(
          ctx, compiler->GetHostToDeviceChannelHandle(channel_name, &channel));

      const auto result_token_tuple = xla::RecvFromHost(
          recv_from_host_token_input, xla_output_shapes->at(i), channel);
      b->ClearOpMetadata();
      recv_from_host_tokens.push_back(
          xla::GetTupleElement(result_token_tuple, /*index=*/1));
      ctx->SetOutput(i, xla::GetTupleElement(result_token_tuple, 0));
    }

    // Set token output.
    xla::XlaOp token_output = recv_from_host_tokens.empty()
                                  ? recv_from_host_token_input
                                  : xla::AfterAll(b, recv_from_host_tokens);
    OP_REQUIRES_OK(
        ctx, ctx->compiler()->SetNodeToken(original_node_name_, token_output));
  }

 private:
  absl::Status LowerFunctionalOps(Graph* g,
                                  const FunctionLibraryDefinition& flib_def) {
    bool modified;
    do {
      modified = false;

      // Lower "If" nodes first. Their body functions will be expanded as
      // function call nodes, which we will lower later.
      // We do not need to lower "While" nodes because shape inference can
      // handle them correctly (output shapes are input shapes).
      std::vector<Node*> if_nodes;
      for (Node* n : g->op_nodes()) {
        if (n->type_string() == "If") {
          if_nodes.push_back(n);
        }
      }
      for (Node* if_node : if_nodes) {
        TF_RETURN_IF_ERROR(
            RewriteIfNode(if_node, g, /*keep_node_fetchable=*/false));
      }
      if (!if_nodes.empty()) {
        modified = true;
      }

      // Lower function call nodes.
      std::vector<Node*> call_nodes;
      for (Node* n : g->op_nodes()) {
        if (IsFunctionCall(flib_def, *n)) {
          call_nodes.push_back(n);
        }
      }
      for (Node* call_node : call_nodes) {
        TF_RETURN_IF_ERROR(RewriteFunctionCallNode(
            call_node, g, flib_def, /*keep_caller_fetchable=*/false));
      }
      if (!call_nodes.empty()) {
        modified = true;
      }
    } while (modified);

    return absl::OkStatus();
  }

  absl::Status InferOutputShapes(XlaOpKernelContext* ctx,
                                 const FunctionLibraryDefinition* flib_def,
                                 std::vector<TensorShape>* output_shapes) {
    // First unpack the inference graphdef from the attr into graph. Don't do
    // any shape inference at this point.
    Graph* graph = shape_inference_graph_function_->graph;

    // Lower functional ops, because they are not friendly to shape inference.
    TF_RETURN_IF_ERROR(LowerFunctionalOps(graph, *flib_def));

    // Now run shape inference, filling in the shapes of recvathost nodes.
    bool got_output_shapes = false;
    ShapeRefiner shape_refiner{graph->versions().producer(),
                               graph->op_registry()};

    // Make sure all nodes can be reached from source node as
    // `GetReversePostOrder` would only collect nodes reachable from source.
    FixupSourceAndSinkEdges(graph);

    std::vector<Node*> nodes;
    GetReversePostOrder(*graph, &nodes);
    for (auto node : nodes) {
      TF_RETURN_IF_ERROR(shape_refiner.AddNode(node));
      if (node->type_string() == kRecvAtHostOp) {
        const AttrValue* key_attr = node->attrs().Find("key");
        if (key_attr == nullptr) {
          return errors::InvalidArgument("Node ", node->name(),
                                         " has no key attribute");
        }
        std::vector<TensorShape> dtoh_shapes;
        if (!ctx->compiler()
                 ->GetDeviceToHostShapes(key_attr->s(), &dtoh_shapes)
                 .ok()) {
          return errors::InvalidArgument(
              "Shape inference for HostCompute ", ctx->op_kernel().name(),
              " failed: host recv node ", node->name(), " with key '",
              key_attr->s(), "' has unknown shapes.");
        }
        if (dtoh_shapes.size() != node->num_outputs()) {
          return errors::InvalidArgument(
              "Shape inference for HostCompute ", ctx->op_kernel().name(),
              " failed: host recv node ", node->name(), " with key '",
              key_attr->s(), "' has ", node->num_outputs(),
              " outputs but inferred shapes expect ", dtoh_shapes.size());
        }
        for (int i = 0; i < node->num_outputs(); ++i) {
          shape_inference::InferenceContext* shape_ctx =
              shape_refiner.GetContext(node);
          shape_inference::ShapeHandle handle;
          TF_RETURN_IF_ERROR(
              shape_ctx->MakeShapeFromTensorShape(dtoh_shapes.at(i), &handle));
          shape_ctx->set_output(i, handle);
        }
      } else if (node->type_string() == kSendFromHostOp) {
        if (got_output_shapes) {
          return errors::InvalidArgument(
              "Shape inference for HostCompute ", ctx->op_kernel().name(),
              " failed: inference graph has multiple send from host nodes");
        }
        got_output_shapes = true;
        // The last input is the dynamic key so don't record its shape.
        output_shapes->resize(node->num_inputs() - 1);
        shape_inference::InferenceContext* shape_ctx =
            shape_refiner.GetContext(node);
        for (int i = 0; i < node->num_inputs() - 1; ++i) {
          shape_inference::ShapeHandle handle = shape_ctx->input(i);
          if (!shape_ctx->FullyDefined(handle)) {
            return errors::InvalidArgument(
                "Shape inference for HostCompute ", ctx->op_kernel().name(),
                " failed: send from host node ", node->name(),
                " has non-fully defined shape of input index ", i);
          }
          TensorShapeProto shape_proto;
          shape_ctx->ShapeHandleToProto(handle, &shape_proto);
          (*output_shapes)[i] = TensorShape(shape_proto);
          VLOG(2) << "Inferred shape " << shape_proto;
        }
      }
    }
    if (!got_output_shapes) {
      return errors::InvalidArgument(
          "Shape inference for HostCompute ", ctx->op_kernel().name(),
          " failed: inference graph has no send from host node");
    }
    return absl::OkStatus();
  }

  DataTypeVector input_dtypes_;
  DataTypeVector output_dtypes_;
  std::vector<string> ancestors_;
  std::vector<TensorShape> static_output_shapes_;
  std::vector<xla::Shape> static_xla_output_shapes_;
  string original_node_name_;
  // If static_xla_output_shapes_.size() == 1 then xla_output_shape_ is the
  // unique output shape, otherwise it is a tuple of all the xla_output_shapes_.
  xla::Shape static_xla_output_shape_;
  string send_key_;
  string recv_key_;
  // If shape inference is performed at runtime, the graph needed to perform
  // shape inference is stored in this function.
  std::unique_ptr<FunctionBody> shape_inference_graph_function_;
  int64_t cost_estimate_;
  int64_t tpu_core_;
  std::vector<string> token_input_nodes_;

  HostComputeOp(const HostComputeOp&) = delete;
  void operator=(const HostComputeOp&) = delete;
};

class SendToHostOp : public XlaOpKernel {
 public:
  explicit SendToHostOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tinput", &input_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kXlaTokenInputNodesAttrName, &token_input_nodes_));
    OP_REQUIRES(ctx, !token_input_nodes_.empty(),
                errors::InvalidArgument("XlaSendToHost node does not have ",
                                        kXlaTokenInputNodesAttrName, " attr"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kXlaOriginalOutsideCompilationNodeName,
                                     &original_node_name_));
  }

  ~SendToHostOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    XlaCompiler* compiler = ctx->compiler();
    xla::XlaOp operand = ctx->Input(0);
    std::vector<xla::XlaOp> input_tokens;
    for (auto& token_input_node : token_input_nodes_) {
      auto token_or = compiler->GetNodeToken(token_input_node);
      OP_REQUIRES_OK(ctx, token_or.status());
      input_tokens.push_back(token_or.value());
    }
    xla::XlaOp token = xla::AfterAll(b, input_tokens);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(input_dtype_, ctx->InputShape(0),
                                              &xla_shape));
    // Specify frontend attributes.
    xla::FrontendAttributes attrs;
    (*attrs.mutable_map())[xla::kXlaHostTransferRendezvousNameAttr] = key_;
    (*attrs.mutable_map())[xla::kXlaHostTransferHandlerNameAttr] =
        xla::kXlaHostTransferTfRendezvousHandlerName;
    b->SetFrontendAttributes(attrs);
    xla::ChannelHandle channel;
    OP_REQUIRES_OK(ctx, compiler->GetDeviceToHostChannelHandle(key_, &channel));
    xla::XlaOp output_token =
        xla::SendToHost(operand, token, xla_shape, channel);
    OP_REQUIRES_OK(ctx,
                   compiler->SetNodeToken(original_node_name_, output_token));
  }

 private:
  DataType input_dtype_;
  string key_;
  std::vector<string> token_input_nodes_;
  string original_node_name_;
  SendToHostOp(const SendToHostOp&) = delete;
  void operator=(const SendToHostOp&) = delete;
};

class RecvFromHostOp : public XlaOpKernel {
 public:
  explicit RecvFromHostOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput", &output_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &output_shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kXlaTokenInputNodesAttrName, &token_input_nodes_));
    OP_REQUIRES(ctx, !token_input_nodes_.empty(),
                errors::InvalidArgument("XlaRecvFromHost node does not have ",
                                        kXlaTokenInputNodesAttrName, " attr"));
    if (!ctx->GetAttr(kXlaOriginalOutsideCompilationNodeName,
                      &original_node_name_)
             .ok())
      original_node_name_ = name();
  }

  ~RecvFromHostOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    XlaCompiler* compiler = ctx->compiler();
    std::vector<xla::XlaOp> input_tokens;
    for (auto& token_input_node : token_input_nodes_) {
      auto token_or = compiler->GetNodeToken(token_input_node);
      OP_REQUIRES_OK(ctx, token_or.status());
      input_tokens.push_back(token_or.value());
    }
    xla::XlaOp token = xla::AfterAll(b, input_tokens);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(
        ctx, TensorShapeToXLAShape(output_dtype_, output_shape_, &xla_shape));
    // Specify frontend attributes.
    xla::FrontendAttributes attrs;
    (*attrs.mutable_map())[xla::kXlaHostTransferRendezvousNameAttr] = key_;
    (*attrs.mutable_map())[xla::kXlaHostTransferHandlerNameAttr] =
        xla::kXlaHostTransferTfRendezvousHandlerName;
    b->SetFrontendAttributes(attrs);
    xla::ChannelHandle channel;
    OP_REQUIRES_OK(ctx, compiler->GetHostToDeviceChannelHandle(key_, &channel));
    xla::XlaOp result = xla::RecvFromHost(token, xla_shape, channel);
    // xla::RecvFromHost returns a tuple of (received data, token).
    ctx->SetOutput(0, xla::GetTupleElement(result, 0));
    OP_REQUIRES_OK(ctx,
                   compiler->SetNodeToken(original_node_name_,
                                          xla::GetTupleElement(result, 1)));
  }

 private:
  DataType output_dtype_;
  TensorShape output_shape_;
  string key_;
  std::vector<string> token_input_nodes_;
  string original_node_name_;
  RecvFromHostOp(const RecvFromHostOp&) = delete;
  void operator=(const RecvFromHostOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaHostCompute"), HostComputeOp);
REGISTER_XLA_OP(Name("XlaSendToHost"), SendToHostOp);
REGISTER_XLA_OP(Name("XlaRecvFromHost"), RecvFromHostOp);
REGISTER_XLA_OP(Name("_XlaHostComputeMlir"), MlirXlaOpKernel);

}  // anonymous namespace
}  // namespace tensorflow
