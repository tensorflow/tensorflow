/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"

#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Parser.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

// Parses the MLIR module from the mlir_module_string.
Status ParseMlirModule(llvm::StringRef mlir_module_string,
                       mlir::MLIRContext* mlir_context,
                       mlir::OwningModuleRef* mlir_module) {
  TF_RET_CHECK(!mlir_module_string.empty())
      << "unexpected empty serialized MLIR module string";
  TF_RET_CHECK(mlir_module) << "unexpected null MLIR module pointer";

  // Parse the module.
  *mlir_module = mlir::parseSourceString(mlir_module_string, mlir_context);
  if (!*mlir_module) {
    return errors::InvalidArgument("could not parse MLIR module");
  }

  return Status::OK();
}

// Converts arg_shapes to xla::Shape's and store into xla_input_shapes.
Status GetXlaInputShapes(
    mlir::ModuleOp module, absl::Span<TensorShape> arg_shapes,
    const xla::CustomShapeRepresentationFn shape_representation_fn,
    std::vector<xla::Shape>* xla_input_shapes) {
  xla_input_shapes->clear();

  mlir::FuncOp main_func = module.lookupSymbol<mlir::FuncOp>("main");
  mlir::FunctionType func_type = main_func.getType();

  int num_args = func_type.getNumInputs();
  xla_input_shapes->reserve(num_args);

  std::vector<xla::Shape> individual_arg_shapes;
  individual_arg_shapes.reserve(num_args);
  for (int i = 0; i < num_args; ++i) {
    individual_arg_shapes.emplace_back();
    xla::Shape& xla_shape = individual_arg_shapes.back();

    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(func_type.getInput(i), &dtype));
    TF_ASSIGN_OR_RETURN(xla_shape,
                        shape_representation_fn(arg_shapes[i], dtype));
  }
  xla_input_shapes->push_back(
      xla::ShapeUtil::MakeTupleShape(individual_arg_shapes));
  return Status::OK();
}

// Calculates computation output shape and build OutputDescription for each
// output based on static shapes in MLIR module
Status GetOutputInfo(
    mlir::ModuleOp module,
    const xla::CustomShapeRepresentationFn shape_representation_fn,
    xla::Shape* xla_output_shape,
    std::vector<XlaCompiler::OutputDescription>* outputs) {
  mlir::FuncOp main_func = module.lookupSymbol<mlir::FuncOp>("main");
  mlir::FunctionType func_type = main_func.getType();

  outputs->clear();
  outputs->reserve(func_type.getNumResults());

  std::vector<xla::Shape> shapes;
  shapes.reserve(func_type.getNumResults());

  for (mlir::Type type : func_type.getResults()) {
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        TypeToShape(type, shape_representation_fn));
    auto tensor_type = type.dyn_cast<mlir::RankedTensorType>();
    shapes.push_back(shape);

    // Construct OutputDescription for result.
    outputs->emplace_back();
    XlaCompiler::OutputDescription& out_desc = outputs->back();
    TF_RETURN_IF_ERROR(ConvertToDataType(tensor_type, &out_desc.type));
    // TODO(ycao): Support constant output.
    out_desc.is_constant = false;
    TF_RETURN_IF_ERROR(XLAShapeToTensorShape(shape, &out_desc.shape));
    // Input_index is only meaningful for resource output. Since MLIR-based
    // TF-Compiler bridge doesn't support resource output yet. Setting it to
    // meaningless value -1.
    // TODO(ycao): Support resource-type output.
    out_desc.input_index = -1;
    // MLIR-based TF-Compiler bridge doesn't support tensorlist output yet.
    // TODO(ycao): Support tensorlist-type output.
    out_desc.is_tensor_list = false;
  }

  // XLA computation always uses Tuple shape.
  *xla_output_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  return Status::OK();
}

// Gets information about how computation updates Tensorflow resources.
// TODO(ycao): Implement logic to compute resource updates when we need to
// support graphs with resource updates in MLIR-based TF compiler bridge.
void GetResourceUpdatesForMlir(
    std::vector<XlaCompiler::ResourceUpdate>* resource_updates) {
  resource_updates->clear();
}

// Creates a vector that maps from the parameters of the XLA computation to
// their original argument positions.
// MLIR-based TF-Compiler bridge doesn't have constant analysis yet, thus no
// inputs are known constants. Therefore, the input mapping between input to
// computation arguments is a trivial in-order 1-1 mapping.
// TODO(ycao): Support computation with compile-time constant, which requires
// non-trivial input mapping as implemented now.
void GetInputMappingForMlir(int num_inputs, std::vector<int>* input_mapping) {
  input_mapping->resize(num_inputs, 0);
  std::iota(input_mapping->begin(), input_mapping->end(), 0);
}

// Refine MLIR types based on new shape information.
Status RefineShapes(absl::Span<TensorShape> arg_shapes, mlir::ModuleOp module) {
  auto versions = module.getAttrOfType<::mlir::DictionaryAttr>("tf.versions");
  if (!versions) {
    return errors::Internal(
        "Missing 'tf.versions' attribute on the module, abort.\n");
  }
  auto producer = versions.get("producer").dyn_cast<mlir::IntegerAttr>();
  if (!producer) {
    return errors::Internal(
        "Missing 'producer' attribute on the module, abort.\n");
  }

  llvm::SmallVector<int64_t, 16> shape_backing;
  llvm::SmallVector<llvm::ArrayRef<int64_t>, 4> arg_shapes_copy;
  {
    // Convert arg_shapes to a mlir friendly format.
    size_t count = 0;
    for (const TensorShape& shape : arg_shapes) {
      count += shape.dims();
    }
    shape_backing.resize(count);
    arg_shapes_copy.reserve(arg_shapes.size());
    size_t offset = 0;
    for (const TensorShape& shape : arg_shapes) {
      size_t start = offset;
      for (tensorflow::TensorShapeDim dim : shape) {
        shape_backing[offset] = dim.size;
        ++offset;
      }
      if (offset == start) {
        arg_shapes_copy.push_back(llvm::ArrayRef<int64_t>());
      } else {
        arg_shapes_copy.push_back(
            llvm::ArrayRef<int64_t>(&shape_backing[start], offset - start));
      }
    }
  }

  auto main_func = module.lookupSymbol<mlir::FuncOp>("main");

  mlir::StatusScopedDiagnosticHandler error_handler(module.getContext());
  mlir::LogicalResult result = mlir::TF::InferShapeForFunction(
      main_func, arg_shapes_copy, producer.getInt());

  if (failed(result)) {
    return error_handler.Combine(
        errors::Internal("MLIR Shape refinement failed"));
  }
  return Status::OK();
}

}  //  namespace

Status ConvertMLIRToXlaComputation(mlir::ModuleOp module_op,
                                   xla::XlaComputation* xla_computation,
                                   bool use_tuple_args,
                                   bool always_return_tuple) {
  {
    // Make sure we catch any error reported by MLIR and forward it to the TF
    // error reporting system. Report a generic error if pass manager failed
    // without emitting a diagnostic.
    mlir::StatusScopedDiagnosticHandler error_handler(module_op.getContext());
    mlir::LogicalResult result = mlir::xla_hlo::legalizeTF(module_op);
    if (failed(result)) {
      return error_handler.Combine(
          errors::Internal("MLIR TF to XLA legalization failed"));
    }
  }

  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("mlir_compile_legalize_hlo", module_op);

  xla::HloProto hlo_proto;
  TF_RETURN_IF_ERROR(mlir::ConvertMlirHloToHlo(
      module_op, &hlo_proto, use_tuple_args, always_return_tuple));
  *xla_computation = xla::XlaComputation(hlo_proto.hlo_module());
  return Status::OK();
}

Status CompileSerializedMlirToXlaHlo(
    llvm::StringRef mlir_module_string, absl::Span<TensorShape> arg_shapes,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    XlaCompiler::CompilationResult* compilation_result) {
  mlir::MLIRContext mlir_context;
  mlir::OwningModuleRef mlir_module;

  TF_RETURN_IF_ERROR(
      ParseMlirModule(mlir_module_string, &mlir_context, &mlir_module));
  auto module_op = mlir_module.get();

  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("mlir_compile_before", module_op);

  // Use arg_shapes to improve the mlir type information of `main` in module_op.
  TF_RETURN_IF_ERROR(RefineShapes(arg_shapes, module_op));

  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("mlir_compile_shape_refiner", module_op);

  // Convert MLIR module to XLA HLO proto contained in XlaComputation.
  compilation_result->computation = std::make_shared<xla::XlaComputation>();
  TF_RETURN_IF_ERROR(ConvertMLIRToXlaComputation(
      module_op, compilation_result->computation.get(), /*use_tuple_args=*/true,
      /*always_return_tuple=*/false));

  // Construct mapping from XlaComputation's arg to input edges of execute
  // node.
  GetInputMappingForMlir(arg_shapes.size(), &compilation_result->input_mapping);

  auto shape_representation_fn_no_fast_memory =
      [shape_representation_fn](const TensorShape& shape, DataType dtype) {
        return shape_representation_fn(shape, dtype, /*use_fast_memory=*/false);
      };

  // Compute all input shapes.
  TF_RETURN_IF_ERROR(GetXlaInputShapes(module_op, arg_shapes,
                                       shape_representation_fn_no_fast_memory,
                                       &compilation_result->xla_input_shapes));

  // Compute all output descriptions.
  TF_RETURN_IF_ERROR(GetOutputInfo(
      module_op, shape_representation_fn_no_fast_memory,
      &compilation_result->xla_output_shape, &compilation_result->outputs));

  // Compute what resource variables need to be updated after XlaComputation's
  // execution.
  GetResourceUpdatesForMlir(&compilation_result->resource_updates);

  if (VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("mlir_compile_after", module_op);

  return Status::OK();
}

}  // namespace tensorflow
