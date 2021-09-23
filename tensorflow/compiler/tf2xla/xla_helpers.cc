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
  return [](const TensorShape& shape, DataType dtype,
            bool use_fast_memory) -> StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype, shape, &xla_shape));
    return xla_shape;
  };
}

// Rewrites the layout of xla_shape if there is tiled sharding.
Status RewriteLayoutWithShardedShape(
    const absl::optional<xla::HloSharding>& sharding, bool use_fast_memory,
    XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    xla::Shape* xla_shape) {
  if (sharding && !sharding->IsTileMaximal() && !sharding->IsManual()) {
    // After sharding, per core shape might have different layout. For example,
    // before sharding, a shape [128, 128] will be assigned default
    // minor-to-major {1, 0}. But after we shard this shape to [128, 64] * 2,
    // the sharded shapes will have minor-to-major {0, 1}.
    //
    // As a result, for sharded shapes, we set their layout to per core shape's
    // layout.
    //
    // TODO(endlessroad): for variable input & update, we might have
    // different layouts which will prevent input output aliasing and
    // increase memory usage. Investigate such cases.
    int64_t device = *sharding->tile_assignment().begin();
    std::vector<int64_t> offset =
        sharding->TileOffsetForDevice(*xla_shape, device);
    std::vector<int64_t> limit =
        sharding->TileLimitForDevice(*xla_shape, device);
    std::vector<int64_t> dimensions(xla_shape->rank());
    for (int64_t i = 0; i < xla_shape->rank(); ++i) {
      dimensions[i] = limit[i] - offset[i];
    }
    xla::Shape per_device_xla_shape =
        xla::ShapeUtil::MakeShape(xla_shape->element_type(), dimensions);
    TensorShape per_device_tensor_shape;
    TF_RETURN_IF_ERROR(
        XLAShapeToTensorShape(per_device_xla_shape, &per_device_tensor_shape));
    TF_ASSIGN_OR_RETURN(DataType dtype, EncodePrimitiveTypeAsDataType(
                                            xla_shape->element_type()));
    TF_ASSIGN_OR_RETURN(per_device_xla_shape,
                        shape_representation_fn(per_device_tensor_shape, dtype,
                                                use_fast_memory));
    *xla_shape->mutable_layout() = per_device_xla_shape.layout();
  }
  return Status::OK();
}

// There is a shape_representation_fn or sharding for an output, this function
// uses a reshape to fix the layout.
StatusOr<xla::XlaOp> ReshapeWithCorrectRepresentationAndSharding(
    xla::XlaBuilder* builder, xla::XlaOp original, xla::Shape original_shape,
    XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    absl::optional<xla::OpSharding> sharding, bool fast_mem) {
  if (original_shape.IsTuple()) {
    std::vector<xla::XlaOp> elements;
    for (int64_t i = 0; i < original_shape.tuple_shapes_size(); ++i) {
      auto subsharding = sharding ? sharding->tuple_shardings(i) : sharding;
      TF_ASSIGN_OR_RETURN(auto element,
                          ReshapeWithCorrectRepresentationAndSharding(
                              builder, xla::GetTupleElement(original, i),
                              original_shape.tuple_shapes(i),
                              shape_representation_fn, subsharding, fast_mem));
      elements.push_back(element);
    }
    return xla::Tuple(builder, elements);
  }
  if (!original_shape.IsArray()) return original;
  TensorShape shape;
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(original_shape, &shape));
  TF_ASSIGN_OR_RETURN(DataType dtype, EncodePrimitiveTypeAsDataType(
                                          original_shape.element_type()));
  TF_ASSIGN_OR_RETURN(auto to_shape,
                      shape_representation_fn(shape, dtype, fast_mem));
  if (sharding) {
    TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                        xla::HloSharding::FromProto(*sharding));
    TF_RETURN_IF_ERROR(RewriteLayoutWithShardedShape(
        hlo_sharding, fast_mem, shape_representation_fn, &to_shape));
  }
  if (xla::ShapeUtil::Compatible(original_shape, to_shape)) {
    for (int64_t i = 0; i < original_shape.rank(); ++i) {
      to_shape.set_dynamic_dimension(i, original_shape.is_dynamic_dimension(i));
    }
  }
  return xla::Reshape(to_shape, original);
}

Status ResolveDeviceAssignment(
    OpKernelContext* ctx,
    const absl::optional<XlaCompilationResult::CollectiveReduceV2OpInfo>&
        collective_reduce_info,
    xla::ExecutableRunOptions& run_options,
    xla::DeviceAssignment& device_assignment,
    xla::gpu::GpuExecutableRunOptions& gpu_options) {
  // TODO(nnigania): workaround for b/199436990
  static const int kTimeoutSeconds = 300;
  if (!collective_reduce_info) {
    // An empty device assignment is sufficient for the case where no
    // collectives are present.
    return Status::OK();
  }
  if (ctx->collective_executor() == nullptr) {
    return errors::InvalidArgument(
        "CollectiveExecutor is required but not available");
  }

  auto params = core::RefCountPtr<CollectiveParams>(new CollectiveParams());
  params->name = "xla-reduction-compilation";
  params->group.device_type =
      DeviceType{static_cast<Device*>(ctx->device())->device_type()};
  params->group.group_size = collective_reduce_info->group_size;
  params->group.group_key = collective_reduce_info->group_key;
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
        const DeviceBase::GpuDeviceInfo* gpu_device_info =
            resolved_device->tensorflow_gpu_device_info();
        global_device_ids[gpu_device_info->stream->parent()->device_ordinal()] =
            device_attributes.xla_global_id();
      }
    }
    gpu_options.set_gpu_global_device_ids(global_device_ids);
  }
  run_options.set_device_assignment(&device_assignment);
  run_options.set_gpu_executable_run_options(&gpu_options);
  return Status::OK();
}

std::string DefinitionLocationMsg(
    const absl::optional<ManagedStackTrace>& stack_trace) {
  if (stack_trace) {
    std::vector<StackFrame> stack_frames =
        stack_trace->ToStackFrames({}, IsInternalFrameForFilename,
                                   /*reverse_traversal=*/true,
                                   /*limit=*/1);
    if (!stack_frames.empty()) {
      const StackFrame& last_frame = stack_frames[0];
      return absl::StrCat(" (defined @ ", last_frame.file_name, ":",
                          last_frame.line_number, ")");
    }
  }
  return "";
}

}  // end namespace tensorflow
