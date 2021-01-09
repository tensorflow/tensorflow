/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace tensorflow {
namespace {

class XlaSortOp : public XlaOpKernel {
 public:
  explicit XlaSortOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    context->SetOutput(0, xla::Sort({context->Input("input")},
                                    xla::CreateScalarLtComputation(
                                        {context->InputXlaType("input")},
                                        context->builder())));
  }
};

REGISTER_XLA_OP(Name("XlaSort"), XlaSortOp);

class XlaKeyValueSortOp : public XlaOpKernel {
 public:
  explicit XlaKeyValueSortOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    xla::XlaOp result = xla::Sort(
        {context->Input("keys"), context->Input("values")},
        xla::CreateScalarLtComputation(
            {context->InputXlaType("keys"), context->InputXlaType("values")},
            context->builder()));
    context->SetOutput(0, xla::GetTupleElement(result, 0));
    context->SetOutput(1, xla::GetTupleElement(result, 1));
  }
};

REGISTER_XLA_OP(Name("XlaKeyValueSort"), XlaKeyValueSortOp);

class XlaVariadicSortOp : public XlaOpKernel {
 public:
  explicit XlaVariadicSortOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("T", &input_types_));
    OP_REQUIRES_OK(context, context->GetAttr("comparator", &comparator_));
    OP_REQUIRES_OK(context, context->GetAttr("is_stable", &is_stable_));
  }

  void Compile(XlaOpKernelContext* context) override {
    std::vector<xla::XlaOp> inputs;
    std::vector<TensorShape> input_shapes;
    OP_REQUIRES_OK(context,
                   context->InputList("inputs", &inputs, &input_shapes));
    int64 dimension;
    OP_REQUIRES_OK(context,
                   context->ConstantInputAsIntScalar("dimension", &dimension));

    std::vector<xla::PrimitiveType> input_xla_types(input_types_.size());
    std::vector<XlaCompiler::Argument> comparator_args(2 * input_types_.size());

    for (int i = 0; i < inputs.size(); ++i) {
      OP_REQUIRES_OK(context, DataTypeToPrimitiveType(input_types_[i],
                                                      &input_xla_types[i]));
      XlaCompiler::Argument comparator_arg;
      comparator_arg.kind = XlaCompiler::Argument::kParameter;
      comparator_arg.type = input_types_[i];
      comparator_arg.shape = TensorShape();
      comparator_args[2 * i] = comparator_arg;
      comparator_args[2 * i + 1] = comparator_arg;
    }

    // Build the comparator function.
    XlaCompiler::CompilationResult comparator;
    XlaCompiler::CompileOptions compile_options;
    compile_options.use_tuple_arg = false;
    compile_options.always_return_tuple = false;
    compile_options.is_entry_computation = false;
    OP_REQUIRES_OK(context, context->compiler()->CompileFunction(
                                compile_options, *comparator_, comparator_args,
                                &comparator));

    xla::Shape expected_comparator_output_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeToXLAShape(DT_BOOL, TensorShape(),
                                         &expected_comparator_output_shape));
    OP_REQUIRES(
        context,
        xla::ShapeUtil::Compatible(comparator.xla_output_shape,
                                   expected_comparator_output_shape),
        errors::InvalidArgument(
            "Invalid output shape of XlaVariadicSort comparator. Expected ",
            xla::ShapeUtil::HumanString(expected_comparator_output_shape),
            " got ", xla::ShapeUtil::HumanString(comparator.xla_output_shape)));

    xla::XlaOp outputs =
        xla::Sort(inputs, *comparator.computation, dimension, is_stable_);

    for (int i = 0; i < input_types_.size(); ++i) {
      xla::XlaOp output_handle =
          (input_types_.size() > 1 ? xla::GetTupleElement(outputs, i)
                                   : outputs);
      context->SetOutput(i, output_handle);
    }
  }

 private:
  DataTypeVector input_types_;
  const NameAttrList* comparator_;
  bool is_stable_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaVariadicSortOp);
};

REGISTER_XLA_OP(Name("XlaVariadicSort").CompileTimeConstantInput("dimension"),
                XlaVariadicSortOp);
}  // namespace
}  // namespace tensorflow
