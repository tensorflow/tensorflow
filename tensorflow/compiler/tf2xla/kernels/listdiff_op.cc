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

// XLA-specific ListDiff Op. This only supports constant DT_INT32 and DT_INT64
// input.

#include <unordered_set>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

constexpr std::array<DataType, 2> kListDiffTypes = {DT_INT32, DT_INT64};

// ListDiffOp is an XLA kernel that supports constant-only x and y input.
class ListDiffOp : public XlaOpKernel {
 public:
  explicit ListDiffOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    OP_REQUIRES(context, TensorShapeUtils::IsVector(context->InputShape(0)),
                errors::InvalidArgument("ListDiff expects x as a vector, not ",
                                        context->InputShape(0).DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(context->InputShape(1)),
                errors::InvalidArgument("ListDiff expects y as a vector, not ",
                                        context->InputShape(1).DebugString()));

    DataType val_type = context->expected_output_dtype(0);
    DataType idx_type = context->expected_output_dtype(1);

    Status status;
    switch (val_type) {
      case DT_INT32:
        status = ListDiffWithIndexType<int32>(context, idx_type);
        break;
      case DT_INT64:
        status = ListDiffWithIndexType<int64>(context, idx_type);
        break;
      default:
        // This should never happen since we restrict this kernel to only match
        // inputs with supported Tensor datatype.
        status = errors::InvalidArgument("ListDiff expects x and y as either ",
                                         "int32 or int64, not ",
                                         DataTypeString(val_type));
    }
    OP_REQUIRES_OK(context, status);
  }

 private:
  template <typename Tval, typename Tidx>
  Status ListDiff(XlaOpKernelContext* context) {
    std::vector<int64> x_input, y_input;
    TF_RETURN_IF_ERROR(context->ConstantInputAsIntVector(0, &x_input));
    TF_RETURN_IF_ERROR(context->ConstantInputAsIntVector(1, &y_input));

    std::unordered_set<Tval> y_input_set;
    y_input_set.reserve(y_input.size());
    for (auto y : y_input) {
      y_input_set.insert(y);
    }

    std::vector<Tval> val_output;
    std::vector<Tidx> idx_output;
    auto x_size = x_input.size();
    for (Tidx i = 0; i < x_size; ++i) {
      if (y_input_set.count(x_input[i]) > 0) {
        continue;
      }
      val_output.push_back(x_input[i]);
      idx_output.push_back(i);
    }

    context->SetOutput(0,
                       xla::ConstantR1<Tval>(context->builder(), val_output));
    context->SetOutput(1,
                       xla::ConstantR1<Tidx>(context->builder(), idx_output));
    return Status::OK();
  }

  template <typename Tval>
  Status ListDiffWithIndexType(XlaOpKernelContext* context, DataType idx_type) {
    switch (idx_type) {
      case DT_INT32:
        return ListDiff<Tval, int32>(context);
      case DT_INT64:
        return ListDiff<Tval, int64>(context);
      default:
        return errors::InvalidArgument(
            "ListDiff expects idx_out as either int32 or int64, not ",
            DataTypeString(idx_type));
    }
  }
};

REGISTER_XLA_OP(Name("ListDiff")
                    .TypeConstraint("T", kListDiffTypes)
                    .CompileTimeConstantInput("x")
                    .CompileTimeConstantInput("y"),
                ListDiffOp);

}  // namespace
}  // namespace tensorflow
