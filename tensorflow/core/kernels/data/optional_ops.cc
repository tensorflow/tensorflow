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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/data/optional_ops.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {
namespace data {
namespace {

static Status OptionalDeviceCopy(
    const OptionalVariant& from, OptionalVariant* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  if (from.has_value()) {
    const std::vector<Tensor>& from_values = from.get_values();
    std::vector<Tensor> to_values;
    to_values.reserve(from_values.size());
    for (const Tensor& t : from_values) {
      if (DMAHelper::CanUseDMA(&t) || t.dtype() == DT_VARIANT) {
        // NOTE(skyewm): we're careful to make sure the lifetime of the 'to'
        // Tensor passed to `copy` (i.e. to_values.back()) is the same as the
        // returned 'to' OptionalVariant. This is because `copy` may spawn async
        // callbacks that don't run until after this function returns and access
        // the 'to' Tensor (e.g. BaseGPUDevice::MaybeCopyTensorToGPU).
        to_values.emplace_back(t.dtype());
        TF_RETURN_IF_ERROR(copy(t, &to_values.back()));
      } else {
        to_values.push_back(t);
      }
    }
    *to = OptionalVariant(std::move(to_values));
  } else {
    *to = from;
  }
  return Status::OK();
}

#define REGISTER_OPTIONAL_COPY(DIRECTION)               \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      OptionalVariant, DIRECTION, OptionalDeviceCopy)

REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(OptionalVariant,
                                       kOptionalVariantTypeName);

}  // namespace

void OptionalNoneOp::Compute(OpKernelContext* ctx) {
  OP_REQUIRES_OK(ctx, WriteOptionalNoneToOutput(ctx, 0));
}

void OptionalFromValueOp::Compute(OpKernelContext* ctx) {
  OpInputList components_input;
  OP_REQUIRES_OK(ctx, ctx->input_list("components", &components_input));
  std::vector<Tensor> components(components_input.begin(),
                                 components_input.end());
  OP_REQUIRES_OK(ctx,
                 WriteOptionalWithValueToOutput(ctx, 0, std::move(components)));
}

void OptionalHasValueOp::Compute(OpKernelContext* ctx) {
  const Tensor* optional_input;
  OP_REQUIRES_OK(ctx, ctx->input("optional", &optional_input));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(optional_input->shape()),
              errors::InvalidArgument(
                  "Input to OptionalHasValue must be a scalar tensor "
                  "containing an OptionalVariant object."));
  const OptionalVariant* optional =
      optional_input->scalar<Variant>()().get<OptionalVariant>();
  OP_REQUIRES(
      ctx, optional != nullptr,
      errors::InvalidArgument(
          "Input to OptionalHasValue must be an OptionalVariant object."));
  Tensor* result;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &result));
  result->scalar<bool>()() = optional->has_value();
}

void OptionalGetValueOp::Compute(OpKernelContext* ctx) {
  const Tensor* optional_input;
  OP_REQUIRES_OK(ctx, ctx->input("optional", &optional_input));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(optional_input->shape()),
              errors::InvalidArgument(
                  "Input to OptionalHasValue must be a scalar tensor "
                  "containing an OptionalVariant object."));
  const OptionalVariant* optional =
      optional_input->scalar<Variant>()().get<OptionalVariant>();
  OP_REQUIRES(
      ctx, optional != nullptr,
      errors::InvalidArgument(
          "Input to OptionalHasValue must be an OptionalVariant object."));
  OP_REQUIRES(
      ctx, optional->has_value(),
      errors::InvalidArgument("The given optional does not have a value."));
  const auto& components = optional->get_values();
  OP_REQUIRES(
      ctx, components.size() == output_types_.size(),
      errors::InvalidArgument("The given optional has ", components.size(),
                              " components, expected ", output_types_.size()));
  for (int i = 0; i < components.size(); ++i) {
    OP_REQUIRES(ctx, components[i].dtype() == output_types_[i],
                errors::InvalidArgument(
                    "The given optional does not match the expected type for "
                    "component ",
                    i, ". Expected: ", DataTypeString(output_types_[i]),
                    ". Actual: ", DataTypeString(components[i].dtype()), "."));
    OP_REQUIRES(ctx, output_shapes_[i].IsCompatibleWith(components[i].shape()),
                errors::InvalidArgument(
                    "The given optional does not match the expected shape "
                    "for component ",
                    i, ". Expected: ", output_shapes_[i].DebugString(),
                    ". Actual: ", components[i].shape().DebugString(), "."));
    ctx->set_output(i, components[i]);
  }
}

Status WriteOptionalWithValueToOutput(OpKernelContext* ctx, int output_index,
                                      std::vector<Tensor> value) {
  OptionalVariant v(std::move(value));
  Tensor* variant_t;
  AllocatorAttributes cpu_alloc;
  cpu_alloc.set_on_host(true);
  TF_RETURN_IF_ERROR(ctx->allocate_output(output_index, TensorShape({}),
                                          &variant_t, cpu_alloc));
  variant_t->scalar<Variant>()() = v;
  return Status::OK();
}

Status WriteOptionalNoneToOutput(OpKernelContext* ctx, int output_index) {
  OptionalVariant v;
  Tensor* variant_t;
  AllocatorAttributes cpu_alloc;
  cpu_alloc.set_on_host(true);
  TF_RETURN_IF_ERROR(ctx->allocate_output(output_index, TensorShape({}),
                                          &variant_t, cpu_alloc));
  variant_t->scalar<Variant>()() = v;
  return Status::OK();
}

namespace {

REGISTER_KERNEL_BUILDER(Name("OptionalNone").Device(DEVICE_CPU).Priority(2),
                        OptionalNoneOp);
REGISTER_KERNEL_BUILDER(Name("OptionalNone").Device(DEVICE_GPU).Priority(1),
                        OptionalNoneOp);
REGISTER_KERNEL_BUILDER(
    Name("OptionalFromValue").Device(DEVICE_CPU).Priority(2),
    OptionalFromValueOp);
REGISTER_KERNEL_BUILDER(
    Name("OptionalFromValue").Device(DEVICE_GPU).Priority(1),
    OptionalFromValueOp);

REGISTER_KERNEL_BUILDER(Name("OptionalHasValue").Device(DEVICE_CPU).Priority(2),
                        OptionalHasValueOp);
REGISTER_KERNEL_BUILDER(Name("OptionalHasValue")
                            .Device(DEVICE_GPU)
                            .HostMemory("has_value")
                            .Priority(1),
                        OptionalHasValueOp);
REGISTER_KERNEL_BUILDER(Name("OptionalGetValue").Device(DEVICE_CPU).Priority(2),
                        OptionalGetValueOp);
REGISTER_KERNEL_BUILDER(Name("OptionalGetValue").Device(DEVICE_GPU).Priority(1),
                        OptionalGetValueOp);

}  // namespace

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_CPU, OptionalVariant,
                                         OptionalZerosLike<CPUDevice>);

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_CPU,
                                          OptionalVariant,
                                          OptionalBinaryAdd<CPUDevice>);

}  // namespace data
}  // namespace tensorflow
