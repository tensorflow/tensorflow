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

// This file defines helper routines for XLA compilation.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"

#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {

xla::XlaOp XlaHelpers::Zero(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return xla::ConstantLiteral(b, xla::LiteralUtil::Zero(type));
}

xla::XlaOp XlaHelpers::One(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return xla::ConstantLiteral(b, xla::LiteralUtil::One(type));
}

xla::XlaOp XlaHelpers::IntegerLiteral(xla::XlaBuilder* b, DataType data_type,
                                      int64_t value) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return ::tensorflow::IntegerLiteral(b, type, value);
}

xla::XlaOp XlaHelpers::FloatLiteral(xla::XlaBuilder* b, DataType data_type,
                                    double value) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return ::tensorflow::FloatLiteral(b, type, value);
}

/* static */ Status XlaHelpers::ReshapeLiteral(
    const xla::Literal& input, absl::Span<const int64_t> dimensions,
    xla::Literal* output) {
  if (input.shape().IsTuple()) {
    return errors::InvalidArgument("ReshapeLiteral does not support tuples.");
  }
  xla::Shape shape =
      xla::ShapeUtil::MakeShape(input.shape().element_type(), dimensions);
  int64_t elements_before = xla::ShapeUtil::ElementsIn(input.shape());
  int64_t elements_after = xla::ShapeUtil::ElementsIn(shape);
  if (elements_before != elements_after) {
    return errors::InvalidArgument(
        "Shapes before and after ReshapeLiteral have different numbers of "
        "elements.");
  }

  *output = input.Clone();
  output->mutable_shape_do_not_use()->Swap(&shape);
  return Status::OK();
}

Status XlaHelpers::OneHot(xla::XlaBuilder* builder, int64_t depth, int axis,
                          DataType index_type, const TensorShape& indices_shape,
                          const xla::XlaOp& indices, const xla::XlaOp& on_value,
                          const xla::XlaOp& off_value, xla::XlaOp* one_hot) {
  // Broadcast the linspace constant across the indices along the new axis,
  // and test equality at each position.
  std::vector<int64_t> broadcast_dims(indices_shape.dims());
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);

  TensorShape output_shape = indices_shape;
  output_shape.InsertDim(axis, depth);
  xla::Shape iota_shape;
  TF_RETURN_IF_ERROR(
      TensorShapeToXLAShape(index_type, output_shape, &iota_shape));

  // Selects the user-provided off_value and on_value values.
  *one_hot = xla::Select(
      xla::Eq(indices, xla::Iota(builder, iota_shape, axis), broadcast_dims),
      xla::Broadcast(on_value, output_shape.dim_sizes()),
      xla::Broadcast(off_value, output_shape.dim_sizes()));
  return Status::OK();
}

DataType XlaHelpers::SumAccumulationType(const DataType& dtype) {
  // Upcast 16 bit sum reductions to 32 bit to reduce the precision loss from
  // repeated floating point additions.
  if (dtype == DT_BFLOAT16 || dtype == DT_HALF) {
    return DT_FLOAT;
  }
  // Upcast small integer types to 32 bit to avoid overflow.
  if (dtype == DT_INT8 || dtype == DT_INT16) {
    return DT_INT32;
  }
  if (dtype == DT_UINT8 || dtype == DT_UINT16) {
    return DT_UINT32;
  }
  return dtype;
}

xla::XlaOp XlaHelpers::ConvertElementType(const xla::XlaOp& operand,
                                          const DataType new_element_type) {
  xla::PrimitiveType convert_to;
  TF_CHECK_OK(DataTypeToPrimitiveType(new_element_type, &convert_to));
  return xla::ConvertElementType(operand, convert_to);
}

XlaHelpers::ShapeRepresentationFn IdentityShapeRepresentationFn() {
  return [](const TensorShape& shape, DataType dtype, bool use_fast_memory,
            XlaLayoutPreference layout_preference) -> StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype, shape, &xla_shape));
    return xla_shape;
  };
}

Status ResolveDeviceAssignment(
    OpKernelContext* ctx,
    const XlaCompilationResult::CollectiveInfo& collective_info,
    xla::ExecutableRunOptions& run_options,
    xla::DeviceAssignment& device_assignment,
    xla::gpu::GpuExecutableRunOptions& gpu_options) {
  // TODO(nnigania): workaround for b/199436990
  static const int kTimeoutSeconds = 1000;
  if (ctx->collective_executor() == nullptr) {
    return errors::InvalidArgument(
        "CollectiveExecutor is required but not available");
  }

  auto params = core::RefCountPtr<CollectiveParams>(new CollectiveParams());
  params->name = "xla-reduction-compilation";
  params->group.device_type =
      DeviceType{static_cast<Device*>(ctx->device())->device_type()};
  params->group.group_size = collective_info.group_size;
  params->group.group_key = collective_info.group_key;
  params->instance.type = REDUCTION_COLLECTIVE;
  params->instance.impl_details.communication_hint = "nccl";
  params->instance.impl_details.timeout_seconds = kTimeoutSeconds;
  params->instance.impl_details.collective_name = "NcclReduce";
  // TODO(cheshire): Avoid passing a dummy shape, TF runtime does not resolve
  // devices otherwise.
  params->instance.shape = TensorShape({1});

  Status st;
  absl::Notification n;
  ctx->collective_executor()->CompleteParamsAsync(
      ctx->device()->attributes(), params.get(), ctx->cancellation_manager(),
      [&](const Status& s) {
        st = s;
        n.Notify();
      });
  if (!n.WaitForNotificationWithTimeout(absl::Seconds(kTimeoutSeconds))) {
    return errors::InvalidArgument("Timeout reached");
  }
  TF_RETURN_IF_ERROR(st);
  VLOG(5) << "Using collective params to resolve device assignment: "
          << params->ToString();

  // Identify the physical device associated with each replica.
  device_assignment = xla::DeviceAssignment(params->group.group_size, 1);
  for (int device_idx = 0; device_idx < params->group.group_size;
       device_idx++) {
    const DeviceAttributes& device = params->group.members[device_idx].device;
    if (device.xla_global_id() == -1) {
      if (params->group.device_type == DEVICE_TPU) {
        return errors::InvalidArgument(
            absl::StrCat("No global ID was set for TPU device ", device.name(),
                         ". Try initializing the TPU system, e.g. "
                         "`tf.tpu.experimental.initialize_tpu_system()`."));
      } else if (params->group.device_type == DEVICE_GPU) {
        return errors::Internal(
            absl::StrCat("No global ID was set for ", device.name(),
                         ". This is unexpected, please file a bug."));
      } else {
        // TODO(b/194942685): Implement CPU collectives.
        return errors::Unimplemented(
            absl::StrCat("Collectives are not yet implemented for ",
                         params->group.device_type.type_string(),
                         " devices when compiling with XLA. Attempted to "
                         "compile a collective running on",
                         device.name(),
                         ". Please comment on b/194942685 or "
                         "file a new bug if you don't have access."));
      }
    }
    VLOG(2) << "Assigning physical id " << device.xla_global_id()
            << " for replica " << device_idx << " (" << device.name() << ")";
    device_assignment(device_idx, 0) = device.xla_global_id();
  }
  VLOG(5) << "Generated device assignment: " << device_assignment.ToString();
  if (params->group.device_type == DEVICE_GPU) {
    // For GPU collectives, `xla_global_id`s are arbitrary integers, and XLA
    // requires a mapping from local device IDs to global device IDs.
    const DeviceMgr* device_mgr = ctx->function_library()->device_mgr();
    std::vector<xla::GlobalDeviceId> global_device_ids(
        device_mgr->NumDeviceType(params->group.device_type.type_string()));

    for (int device_idx = 0; device_idx < params->group.group_size;
         device_idx++) {
      const DeviceAttributes& device_attributes =
          params->group.members[device_idx].device;
      Device* resolved_device = nullptr;
      Status lookup_status =
          device_mgr->LookupDevice(device_attributes.name(), &resolved_device);
      if (lookup_status.ok()) {
        // This is a local device, so include it in the mapping.
        const DeviceBase::AcceleratorDeviceInfo* accelerator_device_info =
            resolved_device->tensorflow_accelerator_device_info();
        global_device_ids[accelerator_device_info->stream->parent()
                              ->device_ordinal()] =
            device_attributes.xla_global_id();
      }
    }
    gpu_options.set_gpu_global_device_ids(global_device_ids);
  }
  run_options.set_device_assignment(&device_assignment);
  run_options.set_gpu_executable_run_options(&gpu_options);
  return Status::OK();
}

}  // end namespace tensorflow
