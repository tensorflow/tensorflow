/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/kernels/infeed_ops.h"

#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/noncopyable_buffer.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"
#include "xla/stream_executor/tpu/tpu_transfer_manager_interface.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"  // IWYU pragma: keep
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/transfer_ops.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef tensorflow::tpu::NoncopyableBuffer LinearizerBuffer;
typedef std::deque<LinearizerBuffer> LinearizerBufferList;

// For the given shape, chooses a layout for infeed on TPU. The returned shape
// has the same dimensions as the original shape, and only the layout is
// changed.
xla::Shape GetTPUInfeedLayout(const xla::Shape& shape) {
  XLA_Shape c_shape;
  XLA_Shape c_infeed_shape;

  ApiConverter::ToC(shape, &c_shape);

  stream_executor::tpu::ExecutorApiFn()->TpuTransferManager_GetInfeedLayoutFn(
      &c_shape, &c_infeed_shape);
  xla::Shape infeed_shape = ApiConverter::FromC(&c_infeed_shape);
  ApiConverter::Destroy(&c_shape);
  ApiConverter::Destroy(&c_infeed_shape);
  return infeed_shape;
}

// Transposes the given tensor using the tensorflow C++ transpose implementation
// to obtain a XLA literal for the host tensor laid out as the given layout. The
// returned tensor is normalized to the dim0major layout -- F32[10,20,30]{2,0,1}
// is returned as F32[20,10,30]{2,1,0}.
absl::StatusOr<Tensor> TransposeTensor(OpKernelContext* ctx,
                                       const Tensor& input_tensor,
                                       const xla::Shape& xla_shape) {
  tsl::profiler::TraceMe trace_me("TransposeTensor", /*level=*/2);
  const int64_t rank = xla_shape.dimensions().size();
  std::vector<int32_t> permutation(rank);
  std::vector<int64_t> transposed_shapes(rank);
  for (int64_t i = 0; i < rank; ++i) {
    permutation[i] = xla_shape.layout().minor_to_major(rank - 1 - i);
    transposed_shapes[i] = xla_shape.dimensions(permutation[i]);
  }

  Tensor transposed_tensor;

  // If this is a trivial transpose (i.e., bitcast), just create an aliased
  // tensor with the transposed shape.
  if (xla::LayoutUtil::IsMonotonicWithDim0Major(
          xla::ShapeUtil::DropDegenerateDimensions(xla_shape).layout())) {
    TensorShape shape;
    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(transposed_shapes, &shape));
    TF_RETURN_IF_ERROR(transposed_tensor.BitcastFrom(
        input_tensor, input_tensor.dtype(), shape));
    return transposed_tensor;
  }

  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  TF_RETURN_IF_ERROR(ctx->allocate_temp(input_tensor.dtype(),
                                        TensorShape(transposed_shapes),
                                        &transposed_tensor, alloc_attr));
  // Eigen Transpose fails with SIGFPE if there is a dimension of size 0.
  if (input_tensor.NumElements() > 0) {
    TF_RETURN_IF_ERROR(DoTranspose<CPUDevice>(ctx->eigen_device<CPUDevice>(),
                                              input_tensor, permutation,
                                              &transposed_tensor));
  }
  return transposed_tensor;
}

absl::StatusOr<bool> GetLayoutOverride(OpKernelConstruction* ctx,
                                       const char* attrn_name,
                                       std::vector<int64_t>* minor_to_major) {
  if (!ctx->HasAttr(attrn_name)) {
    return false;
  }
  TF_RETURN_IF_ERROR(ctx->GetAttr(attrn_name, minor_to_major));
  return !minor_to_major->empty();
}

absl::Status GetInfeedShapeWithLayout(OpKernelConstruction* ctx,
                                      const char* attrn_name,
                                      const xla::Shape& input_shape,
                                      xla::Shape* output_shape) {
  std::vector<int64_t> minor_to_major;
  TF_ASSIGN_OR_RETURN(bool has_override,
                      GetLayoutOverride(ctx, attrn_name, &minor_to_major));
  if (!has_override) {
    *output_shape = input_shape;
    if (output_shape->IsTuple()) {
      int64_t tuple_elements = xla::ShapeUtil::TupleElementCount(*output_shape);
      for (int64_t i = 0; i < tuple_elements; ++i) {
        xla::Shape* sub_shape =
            xla::ShapeUtil::GetMutableSubshape(output_shape, {i});
        *sub_shape->mutable_layout() = GetTPUInfeedLayout(*sub_shape).layout();
      }
    } else {
      *output_shape->mutable_layout() =
          GetTPUInfeedLayout(*output_shape).layout();
    }
    return absl::OkStatus();
  }

  auto layout_func = [](const xla::Shape& shape) -> xla::Layout {
    return GetTPUInfeedLayout(shape).layout();
  };
  return GetShapeWithLayout(input_shape, minor_to_major, layout_func,
                            output_shape);
}

// LinearizedBuffersWrapper is an opaque C++ data structure for the outputs of
// PrelinearizeOp and PrelinearizeTupleOp. It holds the resultant linearized
// buffers and references to input tensors whose underlying storage are shared
// with linearized buffers.
// NOTE: This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot currently serialize an arbitrary
// `LinearizerBufferList` (aka `std::deque<LinearizerBuffer>`)
// object, so the `Encode()` and `Decode()` methods are not implemented.
struct LinearizedBuffersWrapper {
  explicit LinearizedBuffersWrapper() = default;
  explicit LinearizedBuffersWrapper(LinearizerBufferList bufs,
                                    std::vector<tensorflow::Tensor> ts)
      : buffers(std::move(bufs)), tensors(std::move(ts)) {}
  LinearizedBuffersWrapper(const LinearizedBuffersWrapper& wrapper) {
    // tensorflow::Variant requires this copy constructor to compile.
    LOG(FATAL) << "LinearizedBuffersWrapper should not copy.";
  }
  LinearizedBuffersWrapper& operator=(const LinearizedBuffersWrapper& wrapper) =
      delete;
  LinearizedBuffersWrapper(LinearizedBuffersWrapper&&) = default;
  LinearizedBuffersWrapper& operator=(LinearizedBuffersWrapper&&) = default;
  ~LinearizedBuffersWrapper() = default;

  // These functions are tensorflow::Variant requirements.
  string TypeName() const { return "(anonymous)::LinearizedBuffersWrapper"; }
  void Encode(tensorflow::VariantTensorData* data) const {
    LOG(ERROR) << "Encode() is not implemented for LinearizedBuffersWrapper "
                  "objects.";
  }
  bool Decode(const tensorflow::VariantTensorData& data) {
    LOG(ERROR) << "Decode() is not implemented for LinearizedBuffersWrapper "
                  "objects.";
    return false;
  }

  LinearizerBufferList buffers;
  // Save references on tensors whose underlying storage are shared with
  // LiteralLinearizer::Buffer in `buffers`.
  std::vector<tensorflow::Tensor> tensors;
};

absl::Status AutoTransposeAndLinearize(
    OpKernelContext* ctx, const Tensor& input_tensor, const xla::Shape& shape,
    LinearizerBufferList* linearized_buffers,
    std::vector<Tensor>* saved_input_tensors) {
  const Tensor* tensor = &input_tensor;
  // If the given layout is not in dim0major layout, transposes the tensor.
  bool has_transposed = false;
  Tensor transposed_tensor;
  if (!xla::LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    // If the given layout is not in dim0major layout, transpose the tensor.
    TF_ASSIGN_OR_RETURN(transposed_tensor,
                        TransposeTensor(ctx, input_tensor, shape));
    tensor = &transposed_tensor;
    has_transposed = true;
  }

  xla::BorrowingLiteral literal;
  TF_RETURN_IF_ERROR(HostTensorToBorrowingLiteral(*tensor, &literal));

  auto* transfer_manager =
      xla::TpuTransferManagerInterface::GetRegisteredTpuTransferManager();
  TF_RETURN_IF_ERROR(transfer_manager->LinearizeToBuffers(
      literal, transfer_manager->HostShapeToDeviceShape(literal.shape()),
      linearized_buffers));

  // The input tensor is ref-counted. Save a handle on the input tensor if
  // its underlying storage is shared with linearized buffers to prevent
  // input tensor from getting freed.
  for (const auto& buffer : *linearized_buffers) {
    if (!buffer.owns_data() && !has_transposed) {
      // `buffer` is created from zero-copy fast path from the un-transposed
      // input tensor so its underlying data is shared with input tensor.
      // Save a handle to input tensor to increment its ref-count and avoid
      // it getting deallocated after PrelinearizeTupleOp completes.
      saved_input_tensors->push_back(*tensor);
      // A literal can be linearized to zero to two buffers. If any of the
      // linearized buffer shares storage with input tensor. We save exactly
      // one handle on the input tensor.
      break;
    }
  }
  return absl::OkStatus();
}

class StreamExecutorInfeedEnqueueOp : public TpuInfeedEnqueueOp {
 public:
  explicit StreamExecutorInfeedEnqueueOp(OpKernelConstruction* ctx)
      : TpuInfeedEnqueueOp(ctx,
                           std::make_unique<StreamExecutorTransferOpImpl>()) {}

 private:
  StreamExecutorInfeedEnqueueOp(const StreamExecutorInfeedEnqueueOp&) = delete;
  StreamExecutorInfeedEnqueueOp& operator=(
      const StreamExecutorInfeedEnqueueOp&) = delete;
};

class StreamExecutorInfeedEnqueueTupleOp : public TpuInfeedEnqueueTupleOp {
 public:
  explicit StreamExecutorInfeedEnqueueTupleOp(OpKernelConstruction* ctx)
      : TpuInfeedEnqueueTupleOp(
            ctx, std::make_unique<StreamExecutorTransferOpImpl>()) {}

 private:
  StreamExecutorInfeedEnqueueTupleOp(
      const StreamExecutorInfeedEnqueueTupleOp&) = delete;
  StreamExecutorInfeedEnqueueTupleOp& operator=(
      const StreamExecutorInfeedEnqueueTupleOp&) = delete;
};

class StreamExecutorInfeedEnqueuePrelinearizedBufferOp
    : public InfeedEnqueuePrelinearizedBufferOp {
 public:
  explicit StreamExecutorInfeedEnqueuePrelinearizedBufferOp(
      OpKernelConstruction* ctx)
      : InfeedEnqueuePrelinearizedBufferOp(
            ctx, std::make_unique<StreamExecutorTransferOpImpl>()) {}

 private:
  // InfeedEnqueuePrelinearizedBufferOp is neither copyable nor movable.
  StreamExecutorInfeedEnqueuePrelinearizedBufferOp(
      const StreamExecutorInfeedEnqueuePrelinearizedBufferOp&) = delete;
  StreamExecutorInfeedEnqueuePrelinearizedBufferOp& operator=(
      const StreamExecutorInfeedEnqueuePrelinearizedBufferOp&) = delete;
};
}  // anonymous namespace

TpuInfeedEnqueueOp::TpuInfeedEnqueueOp(
    OpKernelConstruction* ctx,
    std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : TpuTransferAsyncOpKernel(ctx, "infeed_enqueue", 8,
                               std::move(transfer_op)) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  xla::Shape shape;
  OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape_, &shape));
  OP_REQUIRES_OK(ctx,
                 GetInfeedShapeWithLayout(ctx, "layout", shape, &xla_shape_));
}

absl::Status TpuInfeedEnqueueOp::DoWork(OpKernelContext* ctx,
                                        int device_ordinal) {
  VLOG(1) << "TpuInfeedEnqueueOp::DoWork. iter_id=" << ctx->frame_iter().iter_id
          << " device_ordinal=" << device_ordinal;
  const Tensor& input_tensor = ctx->input(0);

  // Validate runtime shape and fail if it doesn't match the contract.
  if (input_tensor.dtype() != dtype_) {
    return absl::InvalidArgumentError("Infeed dtype mismatch.");
  }
  if (input_tensor.shape() != shape_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Infeed shape mismatch; expected ", shape_.DebugString(),
                     ", got ", input_tensor.shape().DebugString()));
  }

  const Tensor* tensor = &input_tensor;
  Tensor transposed_tensor;
  if (!xla::LayoutUtil::IsMonotonicWithDim0Major(xla_shape_.layout())) {
    // If the given layout is not in dim0major layout, transpose the tensor.
    TF_ASSIGN_OR_RETURN(transposed_tensor,
                        TransposeTensor(ctx, input_tensor, xla_shape_));
    tensor = &transposed_tensor;
  }

  xla::BorrowingLiteral literal;
  TF_RETURN_IF_ERROR(HostTensorToBorrowingLiteral(*tensor, &literal));

  // Transfer the given literal to the Infeed interface of the device.
  TF_RETURN_IF_ERROR(
      transfer_op_->TransferLiteralToInfeed(device_ordinal, literal));
  VLOG(1) << "TpuInfeedEnqueueOp completes. iter_id="
          << ctx->frame_iter().iter_id << " device_ordinal=" << device_ordinal;
  return absl::OkStatus();
}

TpuInfeedEnqueueTupleOp::TpuInfeedEnqueueTupleOp(
    OpKernelConstruction* ctx,
    std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : TpuTransferAsyncOpKernel(ctx, "infeed_enqueue", 8,
                               std::move(transfer_op)) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
  OP_REQUIRES(
      ctx, shapes_.size() == dtypes_.size(),
      absl::InvalidArgumentError("shapes and dtypes must be the same length."));

  std::vector<xla::Shape> xla_shapes;
  for (int i = 0; i < shapes_.size(); i++) {
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeToXLAShape(dtypes_[i], shapes_[i], &xla_shape));
    xla_shapes.push_back(xla_shape);
  }
  OP_REQUIRES_OK(
      ctx, GetInfeedShapeWithLayout(ctx, "layouts",
                                    xla::ShapeUtil::MakeTupleShape(xla_shapes),
                                    &tuple_shape_));
}

absl::Status TpuInfeedEnqueueTupleOp::DoWork(OpKernelContext* ctx,
                                             int device_ordinal) {
  VLOG(1) << "TpuInfeedEnqueueTupleOp::DoWork. iter_id="
          << ctx->frame_iter().iter_id << " device_ordinal=" << device_ordinal;
  OpInputList values;
  TF_RETURN_IF_ERROR(ctx->input_list("inputs", &values));
  if (values.size() != shapes_.size()) {
    return absl::InvalidArgumentError(
        "Wrong number of inputs to InfeedEnqueueTuple.");
  }

  for (const auto& shapes : shapes_) {
    VLOG(2) << "TransferLiteralToInfeed " << shapes.DebugString();
  }

  std::vector<Tensor> maybe_transposed_tensors;
  maybe_transposed_tensors.reserve(values.size());
  for (int i = 0; i < values.size(); i++) {
    // Validate runtime shapes and fail if it doesn't match the contract.
    const Tensor* tensor = &values[i];
    if (tensor->shape() != shapes_[i]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Infeed shape mismatch for tuple element ", i, "; expected ",
          shapes_[i].DebugString(), ", got ", tensor->shape().DebugString()));
    }
    if (!xla::LayoutUtil::IsMonotonicWithDim0Major(
            tuple_shape_.tuple_shapes(i).layout())) {
      // If the given layout is not in dim0major layout, tranposes the given
      // tensor.
      TF_ASSIGN_OR_RETURN(
          Tensor transposed_tensor,
          TransposeTensor(ctx, *tensor, tuple_shape_.tuple_shapes(i)));
      maybe_transposed_tensors.emplace_back(transposed_tensor);
    } else {
      maybe_transposed_tensors.emplace_back(*tensor);
    }
  }

  xla::BorrowingLiteral tuple;
  TF_RETURN_IF_ERROR(
      HostTensorsToBorrowingLiteralTuple(maybe_transposed_tensors, &tuple));

  // Transfer the given literal to the Infeed interface of the device.
  TF_RETURN_IF_ERROR(
      transfer_op_->TransferLiteralToInfeed(device_ordinal, tuple));

  VLOG(1) << "TpuInfeedEnqueueTupleOp completes. iter_id="
          << ctx->frame_iter().iter_id << " device_ordinal=" << device_ordinal;

  return absl::OkStatus();
}

InfeedEnqueuePrelinearizedBufferOp::InfeedEnqueuePrelinearizedBufferOp(
    OpKernelConstruction* ctx,
    std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : TpuTransferAsyncOpKernel(ctx, "prelinearized_buffers_to_infeed", 8,
                               std::move(transfer_op)) {}
absl::Status InfeedEnqueuePrelinearizedBufferOp::DoWork(OpKernelContext* ctx,
                                                        int device_ordinal) {
  const Tensor& input_tensor = ctx->input(0);
  const LinearizedBuffersWrapper* wrapper =
      input_tensor.scalar<tensorflow::Variant>()()
          .get<LinearizedBuffersWrapper>();
  TF_RETURN_IF_ERROR(
      transfer_op_->TransferBuffersToInfeed(device_ordinal, wrapper->buffers));

  return absl::OkStatus();
}

// These ops execute on either the TPU device or the CPU device. When running on
// CPU they must specify a non-negative value for device_ordinal to indicate
// which TPU to send infeed to.
REGISTER_KERNEL_BUILDER(
    Name("InfeedEnqueue").Device(DEVICE_TPU_NODE).HostMemory("input"),
    StreamExecutorInfeedEnqueueOp);
REGISTER_KERNEL_BUILDER(Name("InfeedEnqueue").Device(DEVICE_CPU),
                        StreamExecutorInfeedEnqueueOp);

REGISTER_KERNEL_BUILDER(
    Name("InfeedEnqueueTuple").Device(DEVICE_TPU_NODE).HostMemory("inputs"),
    StreamExecutorInfeedEnqueueTupleOp);
REGISTER_KERNEL_BUILDER(Name("InfeedEnqueueTuple").Device(DEVICE_CPU),
                        StreamExecutorInfeedEnqueueTupleOp);

// InfeedEnqueuePrelinearizedBuffer op run on CPU and takes a device_ordinal to
// select the right device to infeed.
REGISTER_KERNEL_BUILDER(
    Name("InfeedEnqueuePrelinearizedBuffer").Device(DEVICE_CPU),
    StreamExecutorInfeedEnqueuePrelinearizedBufferOp);

}  // namespace tensorflow
