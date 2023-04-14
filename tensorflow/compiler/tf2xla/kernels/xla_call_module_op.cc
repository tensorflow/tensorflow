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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/compiler/tf2xla/kernels/xla_call_module_loader.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

class XlaCallModuleOp : public XlaOpKernel {
 public:
  explicit XlaCallModuleOp(OpKernelConstruction *ctx) : XlaOpKernel(ctx) {
    int version;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("version", &version));
    string module_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("module", &module_str));
    std::vector<PartialTensorShape> expected_output_shapes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Sout", &expected_output_shapes));
    std::vector<DataType> expected_output_dtypes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &expected_output_dtypes));
    std::vector<string> dim_args_spec;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_args_spec", &dim_args_spec));
    OP_REQUIRES(ctx,
                expected_output_shapes.size() == expected_output_dtypes.size(),
                errors::InvalidArgument("The size of Sout (",
                                        expected_output_shapes.size(),
                                        ") must match the size of Tout (",
                                        expected_output_dtypes.size(), ")"));
    std::vector<string> platforms;
    // Index in platforms of the current platform, or -1 if module does not take
    // a platform index arg.
    int platform_index = -1;
    if (ctx->HasAttr("platforms")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("platforms", &platforms));
      if (!platforms.empty()) {
        string current_device_type = ctx->device_type().type_string();
        string current_platform = "";
        if (current_device_type == DEVICE_CPU_XLA_JIT) {
          current_platform = "CPU";
        } else if (current_device_type == DEVICE_GPU_XLA_JIT) {
#if GOOGLE_CUDA
          current_platform = "CUDA";
#elif TENSORFLOW_USE_ROCM
          current_platform = "ROCM";
#else
          OP_REQUIRES(ctx, false,
                      errors::Unimplemented("CUDA or ROCM build required"));
#endif
        } else if (current_device_type == DEVICE_TPU_XLA_JIT) {
          current_platform = "TPU";
        } else {
          OP_REQUIRES(ctx, false,
                      errors::Unimplemented("Unexpected device type ",
                                            current_device_type));
        }
        VLOG(3) << "Initialized XlaCallModuleOp on " << current_platform;
        auto found_platform =
            std::find(platforms.begin(), platforms.end(), current_platform);
        OP_REQUIRES(ctx, found_platform != platforms.end(),
                    errors::NotFound(
                        "The current platform ", current_platform,
                        " is not among the platforms required by the module: [",
                        absl::StrJoin(platforms, ", "), "]"));
        // We only use a platform index arguments if we support at least 2
        // platforms.
        if (platforms.size() > 1) {
          platform_index = found_platform - platforms.begin();
        }
      }
    }

    auto loader =
        XlaCallModuleLoader::Create(&context_, version, std::move(module_str),
                                    std::move(dim_args_spec), platform_index);
    OP_REQUIRES_OK(ctx, loader.status());
    loader_ = *std::move(loader);
  }

  void Compile(XlaOpKernelContext *ctx) override {
    std::vector<xla::Shape> input_shapes;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      auto shape = ctx->InputXlaShape(i);
      OP_REQUIRES_OK(ctx, shape.status());
      input_shapes.push_back(*std::move(shape));
    }
    OP_REQUIRES_OK(ctx, loader_->RefineDynamicShapes(input_shapes));
    OP_REQUIRES_OK(ctx, loader_->ValidateModule());

    std::vector<xla::XlaOp> inputs(ctx->num_inputs());
    for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
      inputs[i] = ctx->Input(i);
    }

    auto xla_computation = loader_->ToXlaComputation();
    OP_REQUIRES_OK(ctx, xla_computation.status());

    if (VLOG_IS_ON(3)) {
      OP_REQUIRES_VALUE(
          const xla::HloModuleConfig module_config, ctx,
          xla::HloModule::CreateModuleConfigFromProto(
              xla_computation->proto(), xla::GetDebugOptionsFromFlags()));
      OP_REQUIRES_VALUE(std::unique_ptr<xla::HloModule> hlo_module, ctx,
                        xla::HloModule::CreateFromProto(
                            xla_computation->proto(), module_config));
      xla::HloPrintOptions options;
      options = xla::HloPrintOptions::ShortParsable();
      VLOG(3) << "XlaCallModule converted to HLO module "
              << hlo_module->ToString(options);
    }

    xla::XlaOp output = xla::Call(ctx->builder(), *xla_computation, inputs);

    // Check that the resulting computation returns the expected shape
    OP_REQUIRES_VALUE(xla::Shape found_output_shape, ctx,
                      ctx->builder()->GetShape(output));
    VLOG(3) << "XlaCallModule compiled output shape : "
            << xla::ShapeUtil::HumanString(found_output_shape);

    if (loader_->nr_outputs() == 1) {
      ctx->SetOutput(0, output);
    } else {
      for (int i = 0; i < loader_->nr_outputs(); ++i) {
        ctx->SetOutput(i, xla::GetTupleElement(output, i));
      }
    }
  }

 private:
  mlir::MLIRContext context_{mlir::MLIRContext::Threading::DISABLED};
  std::unique_ptr<XlaCallModuleLoader> loader_;
};

REGISTER_XLA_OP(Name("XlaCallModule"), XlaCallModuleOp);

}  // namespace
}  // namespace tensorflow
