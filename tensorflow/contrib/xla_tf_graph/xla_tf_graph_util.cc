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

#include "tensorflow/contrib/xla_tf_graph/xla_tf_graph_util.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace xla_tf_graph {

namespace {

constexpr const char* const GRAPH_NAME = "xla_tf_graph";
constexpr const char* const NODE_NAME_PREFIX = "xla";

Status ConvertPrimitiveTypeToDataType(const xla::PrimitiveType p_type,
                                      DataType* d_type) {
  switch (p_type) {
    case xla::PRED:
      *d_type = DT_BOOL;
      return Status::OK();
    case xla::S8:
      *d_type = DT_INT8;
      return Status::OK();
    case xla::S16:
      *d_type = DT_INT16;
      return Status::OK();
    case xla::S32:
      *d_type = DT_INT32;
      return Status::OK();
    case xla::S64:
      *d_type = DT_INT64;
      return Status::OK();
    case xla::U8:
      *d_type = DT_UINT8;
      return Status::OK();
    case xla::U16:
      *d_type = DT_UINT16;
      return Status::OK();
    case xla::F16:
      *d_type = DT_HALF;
      return Status::OK();
    case xla::F32:
      *d_type = DT_FLOAT;
      return Status::OK();
    case xla::F64:
      *d_type = DT_DOUBLE;
      return Status::OK();
    default:
      return errors::InvalidArgument(
          "Unsupported PrimitiveType in ConvertPrimitiveTypeToDataType ",
          xla::PrimitiveType_Name(p_type));
  }
}

Status ConvertXlaShapeToTensorShapeType(const xla::Shape& xla_shape,
                                        std::vector<TensorShape>* tensor_shapes,
                                        std::vector<DataType>* data_types) {
  switch (xla_shape.element_type()) {
    case xla::TUPLE: {
      for (const xla::Shape& element_shape : xla_shape.tuple_shapes()) {
        if (element_shape.element_type() == xla::TUPLE) {
          return errors::InvalidArgument("Nested tuple is not allowed.");
        }
        TF_RETURN_IF_ERROR(ConvertXlaShapeToTensorShapeType(
            element_shape, tensor_shapes, data_types));
      }
      return Status::OK();
    }
    case xla::PRED:
    case xla::S8:
    case xla::S16:
    case xla::S32:
    case xla::S64:
    case xla::U8:
    case xla::U16:
    case xla::U32:
    case xla::U64:
    case xla::F16:
    case xla::F32:
    case xla::F64: {
      TensorShape shape;
      DataType type;
      TF_RETURN_IF_ERROR(
          ConvertPrimitiveTypeToDataType(xla_shape.element_type(), &type));
      for (const int64& dim : xla_shape.dimensions()) {
        shape.AddDim(dim);
      }
      tensor_shapes->emplace_back(shape);
      data_types->emplace_back(type);
      return Status::OK();
    }
    default:
      return errors::InvalidArgument(
          "Unsupported PrimitiveType in ConvertXlaShapeToTensorShapeType ",
          xla::PrimitiveType_Name(xla_shape.element_type()));
  }
}

string BuildXlaNodeName(const xla::OperationRequest& operation_request,
                        const string& xla_op_type, const string& suffix) {
  const string name = strings::StrCat(
      NODE_NAME_PREFIX, "/", operation_request.output_handle().handle(), "/",
      xla_op_type);
  if (suffix.empty()) {
    return name;
  } else {
    return strings::StrCat(name, "/", suffix);
  }
}

string BuildXlaNodeName(const xla::OperationRequest& operation_request,
                        const string& xla_op_type) {
  return BuildXlaNodeName(operation_request, xla_op_type, "");
}

string BuildXlaNodeOp(const protobuf::Message& msg, const string& suffix) {
  return strings::StrCat(msg.GetDescriptor()->name(), "/", suffix);
}

string BuildXlaNodeOp(const protobuf::Message& msg) {
  return BuildXlaNodeOp(msg, "");
}

Status ConvertOpRequestToXlaNode(const xla::OperationRequest& operation_request,
                                 XlaNode* xla_node) {
  const xla::OpRequest& op_request = operation_request.request();
  switch (op_request.op_case()) {
    case xla::OpRequest::kBinaryOpRequest: {
      const xla::BinaryOpRequest& op = op_request.binary_op_request();
      xla_node->op_type =
          BuildXlaNodeOp(op, xla::BinaryOperation_Name(op.binop()));
      xla_node->name = BuildXlaNodeName(operation_request, xla_node->op_type);
      xla_node->input_ids.emplace_back(std::make_tuple(op.lhs().handle(), 0));
      xla_node->input_ids.emplace_back(std::make_tuple(op.rhs().handle(), 0));
      for (const int64& dim : op.broadcast_dimensions()) {
        xla_node->broadcast_dimensions.emplace_back(dim);
      }
      break;
    }
    case xla::OpRequest::kParameterRequest: {
      const xla::ParameterRequest& op = op_request.parameter_request();
      xla_node->op_type = BuildXlaNodeOp(op, "");
      xla_node->name =
          BuildXlaNodeName(operation_request, xla_node->op_type, op.name());
      break;
    }
    case xla::OpRequest::kVariadicOpRequest: {
      const xla::VariadicOpRequest& op = op_request.variadic_op_request();
      xla_node->op_type =
          BuildXlaNodeOp(op, xla::VariadicOperation_Name(op.varop()));
      xla_node->name = BuildXlaNodeName(operation_request, xla_node->op_type);
      for (const xla::ComputationDataHandle& handle : op.operands()) {
        xla_node->input_ids.emplace_back(std::make_tuple(handle.handle(), 0));
      }
      break;
    }
    case xla::OpRequest::kGetTupleElementRequest: {
      const xla::GetTupleElementRequest& op =
          op_request.get_tuple_element_request();
      xla_node->op_type = BuildXlaNodeOp(op);
      xla_node->name = BuildXlaNodeName(operation_request, xla_node->op_type);
      xla_node->input_ids.emplace_back(
          std::make_tuple(op.operand().handle(), op.index()));
      break;
    }
    default:
      // TODO(satok): Implement all possible cases.
      LOG(FATAL) << "Op request: " << op_request.op_case()
                 << " is not supported yet.";
      break;
  }

  CHECK(!xla_node->name.empty());
  CHECK(!xla_node->op_type.empty());

  TF_RETURN_IF_ERROR(ConvertXlaShapeToTensorShapeType(
      operation_request.output_shape(), &xla_node->output_shapes,
      &xla_node->output_data_types));
  return Status::OK();
}

void SetupXlaCpuClient(std::unique_ptr<FunctionLibraryDefinition>* flib_def,
                       std::unique_ptr<XlaCompiler>* compiler) {
  xla::Client* client = xla::ClientLibrary::LocalClientOrDie();
  XlaOpRegistry::RegisterCompilationKernels();

  FunctionDefLibrary flib;
  flib_def->reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));

  // Setup compiler options
  XlaCompiler::Options options;
  DeviceType device_type(DEVICE_CPU_XLA_JIT);
  options.device_type = &device_type;
  options.flib_def = flib_def->get();
  options.client = client;
  compiler->reset(new XlaCompiler(options));
}

}  // namespace

xla::StatusOr<std::unique_ptr<xla::SessionModule>>
ConvertTfGraphToXlaSessionModule(const std::vector<XlaCompiler::Argument>& args,
                                 std::unique_ptr<Graph> graph) {
  CHECK(graph);

  std::unique_ptr<FunctionLibraryDefinition> flib_def;
  std::unique_ptr<XlaCompiler> compiler;

  SetupXlaCpuClient(&flib_def, &compiler);

  // Compile graph and build computation
  XlaCompiler::CompilationResult result;
  TF_CHECK_OK(compiler->CompileGraph(XlaCompiler::CompileOptions(), GRAPH_NAME,
                                     std::move(graph), args, &result));

  return result.computation->Snapshot();
}

xla::StatusOr<std::unordered_map<int64, XlaNode>>
ConvertXlaSessionModuleToXlaNodes(const xla::SessionModule& session_module) {
  std::unordered_map<int64, XlaNode> xla_nodes;
  for (const auto& operation_request : session_module.entry().requests()) {
    XlaNode xla_node;
    TF_RETURN_IF_ERROR(
        ConvertOpRequestToXlaNode(operation_request.second, &xla_node));
    xla_nodes.emplace(operation_request.first, xla_node);
  }
  return std::move(xla_nodes);
}

}  // namespace xla_tf_graph
}  // namespace tensorflow
