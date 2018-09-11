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
#include "tensorflow/core/kernels/data/optional_ops.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {
namespace data {
namespace {
const char kOptionalVariantTypeName[] = "tensorflow::data::Optional";

// An `OptionalVariant` can represent either an "actual value" (a tuple of
// tensors) or "none", and may be stored in a DT_VARIANT tensor.
class OptionalVariant {
 public:
  // Create an `OptionalVariant` with no actual value.
  OptionalVariant() : values_(nullptr) {}

  // Create an `OptionalVariant` with the actual value given by the tuple of
  // tensors in `values`.
  explicit OptionalVariant(std::vector<Tensor> values)
      : values_(new std::vector<Tensor>(std::move(values))) {}

  OptionalVariant(const OptionalVariant& other) : values_(other.values_) {}

  // Returns true if `this` represents an actual value.
  bool has_value() const { return values_ != nullptr; }

  // REQUIRES: `this->has_value()` must be true.
  const std::vector<Tensor>& get_values() const {
    CHECK(values_) << "Tried to get values from an empty OptionalVariant";
    return *values_;
  }

  // Implementations of the necessary methods for using `OptionalVariant`
  // objects in DT_VARIANT tensors.
  string TypeName() const { return kOptionalVariantTypeName; }
  void Encode(VariantTensorData* data) const {
    data->set_metadata(values_ != nullptr);
    if (values_ != nullptr) {
      for (const auto& t : *values_) {
        *(data->add_tensors()) = t;
      }
    }
  }

  bool Decode(const VariantTensorData& data) {
    if (data.type_name() != TypeName()) {
      return false;
    }
    bool has_value = false;
    if (!data.get_metadata(&has_value)) {
      return false;
    }
    if (has_value) {
      values_.reset(new std::vector<Tensor>(data.tensors()));
    } else {
      values_.reset();
    }
    return true;
  }

  string DebugString() const {
    if (values_) {
      return strings::StrCat("OptionalVariant<", "values: (",
                             str_util::Join(*values_, ", ",
                                            [](string* s, const Tensor& elem) {
                                              *s = elem.DebugString();
                                            }),
                             ")>");
    } else {
      return strings::StrCat("OptionalVariant<None>");
    }
  }

 private:
  std::shared_ptr<const std::vector<Tensor>> values_;
};

class OptionalNoneOp : public OpKernel {
 public:
  explicit OptionalNoneOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, WriteOptionalNoneToOutput(ctx, 0));
  }
};

class OptionalFromValueOp : public OpKernel {
 public:
  explicit OptionalFromValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OpInputList components_input;
    OP_REQUIRES_OK(ctx, ctx->input_list("components", &components_input));
    std::vector<Tensor> components;
    components.reserve(components_input.size());
    for (const Tensor& component_t : components_input) {
      components.push_back(component_t);
    }
    OP_REQUIRES_OK(
        ctx, WriteOptionalWithValueToOutput(ctx, 0, std::move(components)));
  }
};

class OptionalHasValueOp : public OpKernel {
 public:
  explicit OptionalHasValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
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
};

class OptionalGetValueOp : public OpKernel {
 public:
  explicit OptionalGetValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
  }

  void Compute(OpKernelContext* ctx) override {
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
    for (int i = 0; i < components.size(); ++i) {
      OP_REQUIRES(
          ctx, components[i].dtype() == output_types_[i],
          errors::InvalidArgument(
              "The given optional does not match the expected type for "
              "component ",
              i, ". Expected: ", DataTypeString(output_types_[i]),
              ". Actual: ", DataTypeString(components[i].dtype()), "."));
      OP_REQUIRES(ctx,
                  output_shapes_[i].IsCompatibleWith(components[i].shape()),
                  errors::InvalidArgument(
                      "The given optional does not match the expected shape "
                      "for component ",
                      i, ". Expected: ", output_shapes_[i].DebugString(),
                      ". Actual: ", components[i].shape().DebugString(), "."));
      ctx->set_output(i, components[i]);
    }
  }

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("OptionalNone").Device(DEVICE_CPU),
                        OptionalNoneOp);
REGISTER_KERNEL_BUILDER(Name("OptionalNone").Device(DEVICE_GPU),
                        OptionalNoneOp);
REGISTER_KERNEL_BUILDER(Name("OptionalFromValue").Device(DEVICE_CPU),
                        OptionalFromValueOp);
REGISTER_KERNEL_BUILDER(Name("OptionalFromValue").Device(DEVICE_GPU),
                        OptionalFromValueOp);

REGISTER_KERNEL_BUILDER(Name("OptionalHasValue").Device(DEVICE_CPU),
                        OptionalHasValueOp);
REGISTER_KERNEL_BUILDER(
    Name("OptionalHasValue").Device(DEVICE_GPU).HostMemory("has_value"),
    OptionalHasValueOp);
REGISTER_KERNEL_BUILDER(Name("OptionalGetValue").Device(DEVICE_CPU),
                        OptionalGetValueOp);
REGISTER_KERNEL_BUILDER(Name("OptionalGetValue").Device(DEVICE_GPU),
                        OptionalGetValueOp);

static Status OptionalDeviceCopy(
    const OptionalVariant& from, OptionalVariant* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  if (from.has_value()) {
    const std::vector<Tensor>& from_values = from.get_values();
    std::vector<Tensor> to_values;
    to_values.reserve(from_values.size());
    for (const Tensor& t : from_values) {
      if (DMAHelper::CanUseDMA(&t)) {
        Tensor tmp(t.dtype());
        TF_RETURN_IF_ERROR(copy(t, &tmp));
        to_values.push_back(std::move(tmp));
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

#define REGISTER_OPTIONAL_COPY(DIRECTION)                   \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(     \
      OptionalVariant, DIRECTION, kOptionalVariantTypeName, \
      OptionalDeviceCopy)

REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(OptionalVariant,
                                       kOptionalVariantTypeName);

}  // namespace

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

}  // namespace data
}  // namespace tensorflow
