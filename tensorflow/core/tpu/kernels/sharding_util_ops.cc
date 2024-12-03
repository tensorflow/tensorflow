/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#define EIGEN_USE_THREADS

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/tpu/kernels/sharding_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/macros.h"

namespace tensorflow {
namespace {

constexpr absl::string_view kNumSplitsAttrName = "num_splits";
constexpr absl::string_view kNumConcatsAttrName = "num_concats";

absl::Status GetAndValidateAttributesHelper(
    bool split, OpKernelConstruction* ctx, std::vector<int32_t>& num_partitions,
    int& num_slices, std::vector<int32_t>& paddings, bool& has_paddings) {
  absl::string_view num_partitions_attr_name =
      split ? kNumSplitsAttrName : kNumConcatsAttrName;
  TF_RETURN_IF_ERROR(ctx->GetAttr(num_partitions_attr_name, &num_partitions));

  int num_dims_to_split = 0;
  for (int i = 0, e = num_partitions.size(); i < e; ++i) {
    const auto& split = num_partitions[i];
    if (split <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("'", num_partitions_attr_name, "' at index ", i,
                       " must be positive, but got ", split, "."));
    }
    if (split > 1) {
      ++num_dims_to_split;
    }
    num_slices *= split;
  }

  int n;
  TF_RETURN_IF_ERROR(ctx->GetAttr("N", &n));
  if (n != num_slices) {
    return absl::InvalidArgumentError(
        absl::StrCat("'N' must match number of slices ", num_slices, " from '",
                     num_partitions_attr_name, "', but got ", n, "."));
  }

  TF_RETURN_IF_ERROR(ctx->GetAttr("paddings", &paddings));
  const int expected_rank = num_partitions.size();
  if (!paddings.empty()) {
    if (paddings.size() != expected_rank) {
      return absl::InvalidArgumentError(absl::StrCat(
          "'paddings' length must match '", num_partitions_attr_name,
          "' length ", expected_rank, ", but got ", paddings.size(), "."));
    }

    for (int dim = 0; dim < expected_rank; ++dim) {
      if (paddings[dim] < 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("'padding' must be all non-negative, but got ",
                         paddings[dim], " at index ", dim, "."));
      }
      if (paddings[dim] > 0) {
        has_paddings = true;
      }
    }
  } else {
    paddings.assign(expected_rank, 0);
  }

  return absl::OkStatus();
}

void GetAndValidateAttributes(bool split, OpKernelConstruction* ctx,
                              std::vector<int32_t>& num_partitions,
                              int& num_slices, std::vector<int32_t>& paddings,
                              bool& has_paddings) {
  OP_REQUIRES_OK(
      ctx, GetAndValidateAttributesHelper(split, ctx, num_partitions,
                                          num_slices, paddings, has_paddings));
}

absl::string_view kHandle = "handle";
absl::string_view kTensor = "tensor";

template <bool Handle>
absl::Status CreateResourceInvalidDTypeError(const ResourceHandle& handle,
                                             DataType actual_dtype,
                                             DataType expected_dtype) {
  absl::string_view resource_component = Handle ? kHandle : kTensor;
  return absl::InvalidArgumentError(
      absl::StrCat("'T' must match 'resource' variable ", resource_component,
                   " ('", handle.name(), "') container ('", handle.container(),
                   "') dtype ", DataTypeString(actual_dtype), ", but got ",
                   DataTypeString(expected_dtype), "."));
}

constexpr absl::string_view kTensorName = "'input' tensor";
constexpr absl::string_view kResourceName = "'resource' variable tensor";

// Shared base class to save code space
template <typename Device, typename T>
class XlaSplitNDShared : public OpKernel {
 public:
  explicit TF_ATTRIBUTE_NOINLINE XlaSplitNDShared(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    std::vector<int32_t> num_splits;
    int num_slices = 1;
    std::vector<int32_t> paddings;
    bool has_paddings = false;

    GetAndValidateAttributes(/*split=*/true, ctx, num_splits, num_slices,
                             paddings, has_paddings);

    auto xla_nd_splitter = XlaNDSplitter<Device, T>::Create(
        num_splits, num_slices, paddings, has_paddings);
    OP_REQUIRES_OK(ctx, xla_nd_splitter.status());
    splitter_ = *std::move(xla_nd_splitter);
  }

 protected:
  static void TF_ATTRIBUTE_NOINLINE GetDtypeHelper(OpKernelConstruction* ctx,
                                                   const char* attr_name,
                                                   DataType* dtype_ptr) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(attr_name, dtype_ptr));
  }

  std::optional<XlaNDSplitter<Device, T>> splitter_;
};

template <typename Device, typename T>
class XlaSplitNDBaseOp : public XlaSplitNDShared<Device, T> {
 public:
  explicit XlaSplitNDBaseOp(OpKernelConstruction* ctx)
      : XlaSplitNDShared<Device, T>(ctx) {}

 protected:
  void ComputeInternal(
      bool resource, OpKernelContext* ctx,
      const std::function<absl::Status(const Tensor&)>& assign_or_copy_value_fn,
      const Tensor* input) {
    absl::string_view input_name = resource ? kResourceName : kTensorName;
    auto allocate_output_fn = [&](int i, const TensorShape& output_slice_shape,
                                  Tensor** tensor) {
      return ctx->allocate_output(
          /*index=*/i, output_slice_shape, tensor);
    };

    const Device& device = ctx->eigen_device<Device>();
    auto status = this->splitter_->Split(
        input, input_name, assign_or_copy_value_fn, allocate_output_fn, device);
    OP_REQUIRES_OK(ctx, status);
  }
};

template <typename Device, typename T>
class XlaSplitNDOp : public XlaSplitNDBaseOp<Device, T> {
 public:
  explicit TF_ATTRIBUTE_NOINLINE XlaSplitNDOp(OpKernelConstruction* ctx)
      : XlaSplitNDBaseOp<Device, T>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    auto assign_or_copy_value_fn = [&ctx](const Tensor& input) -> absl::Status {
      ctx->set_output(/*index=*/0, input);
      return absl::OkStatus();
    };

    this->ComputeInternal(/*resource=*/false, ctx, assign_or_copy_value_fn,
                          &input);
  }
};

template <typename Device, typename T>
class ReadVariableXlaSplitNDOp : public XlaSplitNDBaseOp<Device, T> {
 public:
  explicit TF_ATTRIBUTE_NOINLINE ReadVariableXlaSplitNDOp(
      OpKernelConstruction* ctx)
      : XlaSplitNDBaseOp<Device, T>(ctx) {
    XlaSplitNDShared<Device, T>::GetDtypeHelper(ctx, "T", &dtype_);
  }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> variable;
    ResourceHandle handle;
    OP_REQUIRES_OK(ctx, HandleFromInput(ctx, 0, &handle));
    const absl::Status status = LookupResource(ctx, handle, &variable);
    OP_REQUIRES(
        ctx, status.ok(),
        absl::InvalidArgumentError(absl::StrCat(
            "'resource' variable handle ('", handle.name(), "') container ('",
            handle.container(), "') cannot be found.")));

    tf_shared_lock ml(*variable->mu());
    const Tensor* input = variable->tensor();
    OP_REQUIRES(
        ctx, input->dtype() == dtype_,
        CreateResourceInvalidDTypeError<false>(handle, input->dtype(), dtype_));

    auto assign_or_copy_value_fn =
        [&ctx, &variable](const Tensor& input) -> absl::Status {
      if (variable->copy_on_read_mode.load()) {
        Tensor* output;
        TF_RETURN_IF_ERROR(
            ctx->allocate_output(/*index=*/0, input.shape(), &output));
        output->flat<T>().device(ctx->eigen_device<Device>()) = input.flat<T>();
      } else {
        ctx->set_output(/*index=*/0, input);
      }
      return absl::OkStatus();
    };

    this->ComputeInternal(/*resource=*/true, ctx, assign_or_copy_value_fn,
                          input);
  }

 private:
  DataType dtype_;
};

#define REGISTER_XLA_SPLIT_ND(type)                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("XlaSplitND").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      XlaSplitNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_XLA_SPLIT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_XLA_SPLIT_ND);
TF_CALL_int4(REGISTER_XLA_SPLIT_ND);
TF_CALL_uint4(REGISTER_XLA_SPLIT_ND);
#undef REGISTER_XLA_SPLIT_ND

#define REGISTER_READ_VARIABLE_XLA_SPLIT_ND(type) \
  REGISTER_KERNEL_BUILDER(                        \
      Name("ReadVariableXlaSplitND")              \
          .Device(DEVICE_CPU)                     \
          .TypeConstraint<type>("T"),             \
      ReadVariableXlaSplitNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_READ_VARIABLE_XLA_SPLIT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_READ_VARIABLE_XLA_SPLIT_ND);
TF_CALL_int4(REGISTER_READ_VARIABLE_XLA_SPLIT_ND);
TF_CALL_uint4(REGISTER_READ_VARIABLE_XLA_SPLIT_ND);
#undef REGISTER_READ_VARIABLE_XLA_SPLIT_ND

// Shared base class to save code space
template <typename Device, typename T>
class XlaConcatNDShared : public OpKernel {
 public:
  explicit TF_ATTRIBUTE_NOINLINE XlaConcatNDShared(OpKernelConstruction* ctx)
      : OpKernel(ctx), num_slices_(1), has_paddings_(false) {
    GetAndValidateAttributes(/*split=*/false, ctx, num_concats_, num_slices_,
                             paddings_, has_paddings_);

    auto xla_nd_concatenator = XlaNDConcatenator<Device, T>::Create(
        num_concats_, num_slices_, paddings_, has_paddings_);
    OP_REQUIRES_OK(ctx, xla_nd_concatenator.status());
    concatenator_ = *std::move(xla_nd_concatenator);
  }

 protected:
  absl::Status GetInputsAndOutputShape(OpKernelContext* ctx,
                                       OpInputList& inputs,
                                       TensorShape& output_shape) {
    TF_RETURN_IF_ERROR(ctx->input_list("inputs", &inputs));
    DCHECK_EQ(inputs.size(), num_slices_);

    const TensorShape& slice_shape = inputs[0].shape();
    if (slice_shape.dims() != num_concats_.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "'inputs' rank must be the same as 'num_concats' length ",
          num_concats_.size(), ", but got rank ", slice_shape.dims(), "."));
    }
    for (int i = 1; i < num_slices_; ++i) {
      const TensorShape& slice_shape_i = inputs[i].shape();
      if (slice_shape != slice_shape_i) {
        return absl::InvalidArgumentError(
            absl::StrCat("'inputs' must all have the same expected shape ",
                         slice_shape.DebugString(), ", but got ",
                         slice_shape_i.DebugString(), " at index ", i, "."));
      }
    }

    for (int i = 0, e = num_concats_.size(); i < e; ++i) {
      const int max_dim_size = slice_shape.dim_size(i) * num_concats_[i];
      if (paddings_[i] > max_dim_size) {
        return absl::InvalidArgumentError(absl::StrCat(
            "'paddings' must not exceed expected output shape dimension ",
            max_dim_size, " at index ", i, ", but got ", paddings_[i], "."));
      }
      TF_RETURN_IF_ERROR(
          output_shape.AddDimWithStatus(max_dim_size - paddings_[i]));
    }

    return absl::OkStatus();
  }

  std::vector<int32_t> num_concats_;
  int num_slices_;
  std::vector<int32_t> paddings_;
  bool has_paddings_;
  std::optional<XlaNDConcatenator<Device, T>> concatenator_;
};

template <typename Device, typename T>
class XlaConcatNDBaseOp : public XlaConcatNDShared<Device, T> {
 public:
  explicit TF_ATTRIBUTE_NOINLINE XlaConcatNDBaseOp(OpKernelConstruction* ctx)
      : XlaConcatNDShared<Device, T>(ctx) {}

 protected:
  void ComputeInternal(
      bool resource, OpKernelContext* ctx, const OpInputList& inputs,
      const std::function<absl::Status(const Tensor&)>& assign_or_copy_value_fn,
      const std::function<absl::StatusOr<Tensor*>()>& get_output_fn) {
    const Device& device = ctx->eigen_device<Device>();
    std::vector<Tensor> input_tensors(inputs.begin(), inputs.end());
    auto status = this->concatenator_->ComputeInternal(
        absl::MakeSpan(input_tensors), assign_or_copy_value_fn, get_output_fn,
        device);
    OP_REQUIRES_OK(ctx, status);
  }
};

template <typename Device, typename T>
class XlaConcatNDOp : public XlaConcatNDBaseOp<Device, T> {
 public:
  explicit XlaConcatNDOp(OpKernelConstruction* ctx)
      : XlaConcatNDBaseOp<Device, T>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OpInputList inputs;
    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   this->GetInputsAndOutputShape(ctx, inputs, output_shape));

    auto assign_or_copy_value_fn = [&ctx](const Tensor& input) -> absl::Status {
      ctx->set_output(/*index=*/0, input);
      return absl::OkStatus();
    };

    auto get_output_fn = [&ctx, &output_shape]() -> absl::StatusOr<Tensor*> {
      Tensor* output = nullptr;
      TF_RETURN_IF_ERROR(
          ctx->allocate_output(/*index=*/0, output_shape, &output));
      return output;
    };
    this->ComputeInternal(/*resource=*/false, ctx, inputs,
                          assign_or_copy_value_fn, get_output_fn);
  }
};

template <typename Device, typename T>
class AssignVariableXlaConcatNDOp : public XlaConcatNDBaseOp<Device, T> {
 public:
  explicit TF_ATTRIBUTE_NOINLINE AssignVariableXlaConcatNDOp(
      OpKernelConstruction* ctx)
      : XlaConcatNDBaseOp<Device, T>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList inputs;
    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   this->GetInputsAndOutputShape(ctx, inputs, output_shape));

    core::RefCountPtr<Var> variable;
    ResourceHandle handle;
    OP_REQUIRES_OK(ctx, HandleFromInput(ctx, 0, &handle));
    if (handle.dtypes_and_shapes().size() == 1) {
      const DtypeAndPartialTensorShape dtype_and_shape =
          handle.dtypes_and_shapes().front();
      OP_REQUIRES(ctx, dtype_and_shape.dtype == dtype_,
                  CreateResourceInvalidDTypeError<true>(
                      handle, dtype_and_shape.dtype, dtype_));
      OP_REQUIRES(ctx, dtype_and_shape.shape.IsCompatibleWith(output_shape),
                  absl::InvalidArgumentError(absl::StrCat(
                      "'resource' variable handle ('", handle.name(),
                      "') container ('", handle.container(),
                      "') shape must be compatible with expected shape ",
                      output_shape.DebugString(), ", but got ",
                      dtype_and_shape.shape.DebugString(), ".")));
    }
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(ctx, handle, &variable,
                                                    [this](Var** ptr) {
                                                      *ptr = new Var(dtype_);
                                                      return absl::OkStatus();
                                                    }));
    mutex_lock ml(*variable->mu());

    OP_REQUIRES(ctx, variable->tensor()->dtype() == dtype_,
                CreateResourceInvalidDTypeError<false>(
                    handle, variable->tensor()->dtype(), dtype_));

    auto assign_or_copy_value_fn = [this, &ctx, &output_shape, &variable](
                                       const Tensor& input) -> absl::Status {
      if (variable->copy_on_read_mode.load()) {
        TF_RETURN_IF_ERROR(
            ctx->allocate_temp(dtype_, output_shape, variable->tensor()));
        variable->tensor()->flat<T>().device(ctx->eigen_device<Device>()) =
            input.flat<T>();
      } else {
        *variable->tensor() = input;
      }
      return absl::OkStatus();
    };

    auto get_output_fn = [this, &ctx, &output_shape,
                          &variable]() -> absl::StatusOr<Tensor*> {
      if (variable->copy_on_read_mode.load() ||
          !variable->tensor()->RefCountIsOne() ||
          !variable->tensor()->shape().IsSameSize(output_shape)) {
        TF_RETURN_IF_ERROR(
            ctx->allocate_temp(dtype_, output_shape, variable->tensor()));
      }
      return variable->tensor();
    };

    this->ComputeInternal(/*resource=*/true, ctx, inputs,
                          assign_or_copy_value_fn, get_output_fn);
    variable->is_initialized = true;
  }

  DataType dtype_;
};

#define REGISTER_XLA_CONCAT_ND(type)                                    \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("XlaConcatND").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      XlaConcatNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_XLA_CONCAT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_XLA_CONCAT_ND);
TF_CALL_int4(REGISTER_XLA_CONCAT_ND);
TF_CALL_uint4(REGISTER_XLA_CONCAT_ND);
#undef REGISTER_XLA_CONCAT_ND

#define REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND(type) \
  REGISTER_KERNEL_BUILDER(                           \
      Name("AssignVariableXlaConcatND")              \
          .Device(DEVICE_CPU)                        \
          .TypeConstraint<type>("T"),                \
      AssignVariableXlaConcatNDOp<Eigen::ThreadPoolDevice, type>)

TF_CALL_POD_TYPES(REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND);
TF_CALL_QUANTIZED_TYPES(REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND);
TF_CALL_int4(REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND);
TF_CALL_uint4(REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND);
#undef REGISTER_ASSIGN_VARIABLE_XLA_CONCAT_ND

}  // anonymous namespace
}  // namespace tensorflow
