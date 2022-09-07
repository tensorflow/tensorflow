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

#include <string>
#include <vector>

#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/stablehlo/stablehlo/dialect/ChloOps.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

void RefineDynamicShapes(XlaOpKernelContext *ctx, mlir::MLIRContext *context,
                         mlir::OwningOpRef<mlir::ModuleOp> *module,
                         int nr_dim_args, bool *dim_args_are_i64);

void PopulateDimArgInputs(XlaOpKernelContext *ctx,
                          std::vector<string> dim_args_spec,
                          bool dim_args_are_i64,
                          std::vector<xla::XlaOp> *inputs);

class XlaCallModuleOp : public XlaOpKernel {
 public:
  explicit XlaCallModuleOp(OpKernelConstruction *ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("module", &module_str_));
    std::vector<PartialTensorShape> expected_output_shapes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Sout", &expected_output_shapes));
    std::vector<DataType> expected_output_dtypes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &expected_output_dtypes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_args_spec", &dim_args_spec_));
    OP_REQUIRES(ctx,
                expected_output_shapes.size() == expected_output_dtypes.size(),
                errors::InvalidArgument("The size of Sout (",
                                        expected_output_shapes.size(),
                                        ") must match the size of Tout (",
                                        expected_output_dtypes.size(), ")"));
    expected_nr_outputs_ = expected_output_shapes.size();
  }

  void Compile(XlaOpKernelContext *ctx) override {
    // Code inpired by
    // tensorflow/compiler/xla/python/mlir.cc::PyMlirModuleToXlaComputation
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::mhlo::MhloDialect>();
    context.loadDialect<mlir::chlo::ChloDialect>();
    context.loadDialect<mlir::TF::TensorFlowDialect>();
    module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(module_str_), &context);
    OP_REQUIRES(ctx, module,
                errors::InvalidArgument("Cannot deserialize MHLO computation"));
    if (failed(module->verifyInvariants())) {
      VLOG(1) << "MLIR verification failed.";
      module->dump();
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Error verifying MHLO module"));
    }

    int nr_dim_args = dim_args_spec_.size();
    std::vector<xla::XlaOp> inputs(nr_dim_args + ctx->num_inputs());

    if (nr_dim_args > 0) {
      bool dim_args_are_i64 = true;
      RefineDynamicShapes(ctx, &context, &module, nr_dim_args,
                          &dim_args_are_i64);
      PopulateDimArgInputs(ctx, dim_args_spec_, dim_args_are_i64, &inputs);
    }
    for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
      inputs[nr_dim_args + i] = ctx->Input(i);
    }

    xla::XlaComputation xla_computation;
    OP_REQUIRES_OK(
        ctx, MlirToXlaComputation(*module, xla_computation, false, false));
    xla::XlaOp output = xla::Call(ctx->builder(), xla_computation, inputs);

    // Check that the resulting computation returns the expected shape
    OP_REQUIRES_VALUE(xla::Shape found_output_shape, ctx,
                      ctx->builder()->GetShape(output));
    VLOG(3) << "XlaCallModule compiled output shape : "
            << xla::ShapeUtil::HumanString(found_output_shape);

    if (expected_nr_outputs_ == 1) {
      ctx->SetOutput(0, output);
    } else {
      for (int i = 0; i < expected_nr_outputs_; ++i) {
        ctx->SetOutput(i, xla::GetTupleElement(output, i));
      }
    }
  }

 private:
  string module_str_;
  int expected_nr_outputs_;
  std::vector<string> dim_args_spec_;
};

// If there are dynamic shapes then resolve the unknown dimensions based on
// the static shapes of the actual arguments and shape inference.
void RefineDynamicShapes(XlaOpKernelContext *ctx, mlir::MLIRContext *context,
                         mlir::OwningOpRef<mlir::ModuleOp> *module,
                         int nr_dim_args, bool *dim_args_are_i64) {
  // Locate the 'main' function.
  // This is the convention used by MlirToXlaComputation.
  auto main = (*module)->lookupSymbol<mlir::func::FuncOp>("main");
  OP_REQUIRES(ctx, main,
              errors::InvalidArgument("Cannot find 'main' in MHLO module"));
  VLOG(3) << "XlaCallModule main function: " << debugString(main);
  mlir::Block &main_body = main.front();

  OP_REQUIRES(ctx,
              nr_dim_args + ctx->num_inputs() == main_body.getNumArguments(),
              errors::InvalidArgument(
                  "Incorrect number of arguments for XlaCallModule. ",
                  "The module expects ", main_body.getNumArguments(),
                  " and dim_args_spec specifies ", nr_dim_args,
                  " dimension arguments, but there are ", ctx->num_inputs(),
                  " actual arguments"));
  // Obtain static input types in MLIR terms.
  mlir::Builder builder(context);

  std::vector<mlir::Type> static_input_types(main_body.getNumArguments());
  // The dim_arg parameters already have known types.
  for (int i = 0; i < nr_dim_args; ++i) {
    static_input_types[i] = getElementTypeOrSelf(main_body.getArgument(i));
    *dim_args_are_i64 = (static_input_types[i].getIntOrFloatBitWidth() == 64);
  }

  // Now the actual arguments
  for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
    OP_REQUIRES_VALUE(xla::Shape xla_shape, ctx, ctx->InputXlaShape(i));
    std::vector<int64_t> xla_dimensions(xla_shape.dimensions().begin(),
                                        xla_shape.dimensions().end());
    OP_REQUIRES_VALUE(
        mlir::Type element_type, ctx,
        ConvertPrimitiveTypeToMLIRType(xla_shape.element_type(), builder));
    mlir::Type type = mlir::RankedTensorType::get(xla_dimensions, element_type);
    // TODO(burmako): This fails with an obscure compilation error.
    // OP_REQUIRES_VALUE(
    //     mlir::Type type, ctx,
    //     ConvertShapeToType<mlir::RankedTensorType>(xla_shape, builder));
    VLOG(3) << "XlaCallModule static input type #" << nr_dim_args + i << ": "
            << debugString(type);
    static_input_types[nr_dim_args + i] = type;
  }

  // Refine 'main' argument types to use static input types instead.
  // This will only change the argument types and will not propagate the
  // additional type information further. For that, we'll need to run
  // shape inference as explained below.
  main.setType(
      builder.getFunctionType(static_input_types, main->getResultTypes()));
  for (auto i = 0; i < main_body.getNumArguments(); ++i) {
    main_body.getArgument(i).setType(static_input_types[i]);
  }

  // --tf-shape-inference, despite its TF-specific name, seems to be general
  // enough to also work on MHLO. (Although it fails if it doesn't see a
  // tf.versions attribute on the module, which we hackily attach).
  auto tf_producer =
      builder.getNamedAttr("producer", builder.getI32IntegerAttr(0));
  (**module)->setAttr("tf.versions", builder.getDictionaryAttr({tf_producer}));

  // Run --tf-shape-inference.
  mlir::PassManager pm(context);
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  OP_REQUIRES(ctx, mlir::succeeded(pm.run(**module)),
              errors::InvalidArgument("MHLO shape inference failed"));
  VLOG(3) << "XlaCallModule main function with inferred types: "
          << debugString(*main);
}

// Compute the dim_arg inputs based on the static shapes of the actual arguments
// and put them in the inputs vector.
void PopulateDimArgInputs(XlaOpKernelContext *ctx,
                          std::vector<string> dim_args_spec,
                          bool dim_args_are_i64,
                          std::vector<xla::XlaOp> *inputs) {
  int nr_dim_args = dim_args_spec.size();
  for (int i = 0; i < nr_dim_args; ++i) {
    string dim_arg_spec = dim_args_spec[i];
    size_t dot_pos = dim_arg_spec.find('.');
    OP_REQUIRES(
        ctx, dot_pos != string::npos && dot_pos + 1 < dim_arg_spec.size(),
        errors::InvalidArgument("Cannot parse dim_args_spec ", dim_arg_spec));
    int arg_idx = std::stoi(dim_arg_spec.substr(0, dot_pos));
    int arg_axis_idx = std::stoi(
        dim_arg_spec.substr(dot_pos + 1, dim_arg_spec.size() - dot_pos));
    OP_REQUIRES_VALUE(xla::Shape xla_shape, ctx, ctx->InputXlaShape(arg_idx));

    int64_t dim_arg_val = xla_shape.dimensions()[arg_axis_idx];
    VLOG(3) << "XlaCallModule dim_input[" << i << "] = " << dim_arg_val;
    if (dim_args_are_i64) {
      (*inputs)[i] = xla::ConstantR0<int64_t>(ctx->builder(), dim_arg_val);
    } else {
      (*inputs)[i] = xla::ConstantR0<int32_t>(ctx->builder(), dim_arg_val);
    }
  }
}

REGISTER_XLA_OP(Name("XlaCallModule"), XlaCallModuleOp);
}  // namespace
}  // namespace tensorflow
