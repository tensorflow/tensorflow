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

#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/mlir/tfrt/function/function.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/runtime/op_logger.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_selector.h"
#include "tensorflow/core/tfrt/eager/virtual_device.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/attribute_utils.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/location.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/metrics/common_metrics.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor_view.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_serialize_utils.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_type_registration.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

namespace {

using tensorflow::down_cast;

constexpr char kGpuDeviceName[] = "GPU";
constexpr char kEnableGrapplerAttr[] = "TFRT_TEST_enable_grappler";

TensorMetadata CreateMetadata(DType dtype, absl::Span<const Index> dim_sizes) {
  return TensorMetadata(
      DType(dtype),
      TensorShape(llvm::ArrayRef<Index>(
          reinterpret_cast<const Index*>(dim_sizes.data()), dim_sizes.size())));
}

tensorflow::DataType ConvertDType(DType kind) {
  switch (kind) {
    case DType::UI8:
      return tensorflow::DT_UINT8;
    case DType::UI16:
      return tensorflow::DT_UINT16;
    case DType::UI32:
      return tensorflow::DT_UINT32;
    case DType::UI64:
      return tensorflow::DT_UINT64;
    case DType::I8:
      return tensorflow::DT_INT8;
    case DType::I16:
      return tensorflow::DT_INT16;
    case DType::I32:
      return tensorflow::DT_INT32;
    case DType::I64:
      return tensorflow::DT_INT64;
    case DType::BF16:
      return tensorflow::DT_BFLOAT16;
    case DType::F16:
      return tensorflow::DT_HALF;
    case DType::F32:
      return tensorflow::DT_FLOAT;
    case DType::F64:
      return tensorflow::DT_DOUBLE;
    case DType::I1:
      return tensorflow::DT_BOOL;
    case DType::Complex64:
      return tensorflow::DT_COMPLEX64;
    case DType::Complex128:
      return tensorflow::DT_COMPLEX128;
    case DType::String:
      return tensorflow::DT_STRING;
    case DType::Resource:
      return tensorflow::DT_RESOURCE;
    case DType::Variant:
      return tensorflow::DT_VARIANT;
    case DType::QUI8:
      return tensorflow::DT_QUINT8;
    case DType::QUI16:
      return tensorflow::DT_QUINT16;
    case DType::QI8:
      return tensorflow::DT_QINT8;
    case DType::QI16:
      return tensorflow::DT_QINT16;
    case DType::QI32:
      return tensorflow::DT_QINT32;
    default:
      LOG(ERROR) << "Unsupported kind " << kind;
      return tensorflow::DT_INVALID;
  }
}

DType ConvertDType(tensorflow::DataType dtype) {
  switch (dtype) {
    case tensorflow::DT_UINT8:
      return static_cast<DType>(DType::UI8);
    case tensorflow::DT_UINT16:
      return static_cast<DType>(DType::UI16);
    case tensorflow::DT_UINT32:
      return static_cast<DType>(DType::UI32);
    case tensorflow::DT_UINT64:
      return static_cast<DType>(DType::UI64);
    case tensorflow::DT_INT8:
      return static_cast<DType>(DType::I8);
    case tensorflow::DT_INT16:
      return static_cast<DType>(DType::I16);
    case tensorflow::DT_INT32:
      return static_cast<DType>(DType::I32);
    case tensorflow::DT_INT64:
      return static_cast<DType>(DType::I64);
    case tensorflow::DT_BFLOAT16:
      return static_cast<DType>(DType::BF16);
    case tensorflow::DT_HALF:
      return static_cast<DType>(DType::F16);
    case tensorflow::DT_FLOAT:
      return static_cast<DType>(DType::F32);
    case tensorflow::DT_DOUBLE:
      return static_cast<DType>(DType::F64);
    case tensorflow::DT_BOOL:
      return static_cast<DType>(DType::I1);
    case tensorflow::DT_STRING:
      return static_cast<DType>(DType::String);
    case tensorflow::DT_COMPLEX64:
      return static_cast<DType>(DType::Complex64);
    case tensorflow::DT_COMPLEX128:
      return static_cast<DType>(DType::Complex128);
    case tensorflow::DT_RESOURCE:
      return static_cast<DType>(DType::Resource);
    case tensorflow::DT_VARIANT:
      return static_cast<DType>(DType::Variant);
    case tensorflow::DT_QUINT8:
      return static_cast<DType>(DType::QUI8);
    case tensorflow::DT_QUINT16:
      return static_cast<DType>(DType::QUI16);
    case tensorflow::DT_QINT8:
      return static_cast<DType>(DType::QI8);
    case tensorflow::DT_QINT16:
      return static_cast<DType>(DType::QI16);
    case tensorflow::DT_QINT32:
      return static_cast<DType>(DType::QI32);
    default:
      LOG(FATAL) << "Unsupported dtype " << dtype;
  }
}

OpAttrType ConvertDTypeToOpAttrType(tensorflow::DataType dtype) {
  switch (dtype) {
    case tensorflow::DT_UINT8:
      return OpAttrType::UI8;
    case tensorflow::DT_UINT16:
      return OpAttrType::UI16;
    case tensorflow::DT_UINT32:
      return OpAttrType::UI32;
    case tensorflow::DT_UINT64:
      return OpAttrType::UI64;
    case tensorflow::DT_INT8:
      return OpAttrType::I8;
    case tensorflow::DT_INT16:
      return OpAttrType::I16;
    case tensorflow::DT_INT32:
      return OpAttrType::I32;
    case tensorflow::DT_INT64:
      return OpAttrType::I64;
    case tensorflow::DT_BFLOAT16:
      return OpAttrType::BF16;
    case tensorflow::DT_HALF:
      return OpAttrType::F16;
    case tensorflow::DT_FLOAT:
      return OpAttrType::F32;
    case tensorflow::DT_DOUBLE:
      return OpAttrType::F64;
    case tensorflow::DT_BOOL:
      return OpAttrType::BOOL;
    case tensorflow::DT_COMPLEX64:
      return OpAttrType::COMPLEX64;
    case tensorflow::DT_COMPLEX128:
      return OpAttrType::COMPLEX128;
    default:
      LOG(FATAL) << "Unsupported dtype " << dtype;
  }
}

// This method will first look at the calling op attrs and then look at the
// function def attrs to find the attribute value.
void GetFuncAttr(const OpAttrs& op_attrs, const std::string& op_name,
                 const tensorflow::FunctionLibraryDefinition& func_lib_def,
                 string_view attr_name, bool* value) {
  bool success = op_attrs.Get(attr_name, value);
  if (success) {
    DVLOG(2) << "Caller explicitly specifies " << attr_name.str()
             << (value ? "=true " : "=false, ");
    return;
  }

  const tensorflow::FunctionDef* function_def = func_lib_def.Find(op_name);
  if (function_def == nullptr) {
    return;
  }

  tensorflow::Status status =
      GetNodeAttr(tensorflow::AttrSlice(&function_def->attr()),
                  {attr_name.data(), attr_name.size()}, value);
  if (status.ok()) {
    DVLOG(2) << "Function definition explicitly specifies " << attr_name.str()
             << (value ? "=true" : "=false");
    return;
  }
}

int64_t GetNextLocationId() {
  static std::atomic<int64_t> id(0);
  return id.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

tensorflow::DataType TensorInterface::Type() const {
  auto kind = tensor_.get().metadata().dtype;
  if (kind == DType::Unsupported) {
    assert(llvm::isa<tensorflow::tfd::RuntimeFallbackTensor>(tensor_.get()));
    return tensor_.get<tensorflow::tfd::RuntimeFallbackTensor>()
        .GetTensorHandle()
        ->DataType();
  }
  return ConvertDType(kind);
}

int TensorInterface::NumDims() const { return tensor_.get().shape().GetRank(); }

int64_t TensorInterface::Dim(int dim_index) const {
  return tensor_.get().shape().GetDimensionSize(dim_index);
}

int64_t TensorInterface::NumElements() const {
  if (!tensor_) {
    return static_cast<int64_t>(tf_tensor_.NumElements());
  }
  return tensor_.get().shape().GetNumElements();
}

size_t TensorInterface::ByteSize() const {
  return tensor_.get().metadata().GetHostSizeInBytes();
}

void* TensorInterface::Data() const {
  if (!tensor_) {
    return tensorflow::TensorCApi::Buffer(tf_tensor_)->data();
  } else {
    auto& tensor = tensor_.get<DenseHostTensor>();
    return tensor.data();
  }
}

// TFRT DenseHostTensor is always aligned
bool TensorInterface::IsAligned() const { return true; }

bool TensorInterface::CanMove() const {
  // It is safe to move the Tensor if and only if we own the unique reference to
  // the tensor buffer.
  auto& dht = tensor_.get<DenseHostTensor>();
  return tensor_.IsUnique() && dht.buffer()->IsUnique();
}

std::string TensorInterface::SummarizeValue() const {
  if (!tensor_) {
    return tf_tensor_.SummarizeValue(/*max_entries=*/3, /*print_v2=*/true);
  } else {
    std::string result;
    llvm::raw_string_ostream result_ostream(result);
    tensor_->Print(result_ostream);
    return result;
  }
}

AsyncValueRef<Tensor> TensorInterface::TensorRef() const {
  return tensor_.CopyRef();
}

TensorHandleInterface::TensorHandleInterface(Value&& v, TfrtContext* context)
    : ImmediateExecutionTensorHandle(kTfrt),
      context_(*context),
      value_(std::move(v)) {}

TensorHandleInterface::TensorHandleInterface(tensorflow::DataType dtype,
                                             Value&& v, TfrtContext* context)
    : ImmediateExecutionTensorHandle(kTfrt),
      dtype_(dtype),
      context_(*context),
      value_(std::move(v)) {}

tensorflow::DataType TensorHandleInterface::DataType() const {
  // If dtype_ field is set, use it instead of waiting for the underlying
  // TensorHandle's metadata to be available.
  if (dtype_) {
    return dtype_.value();
  }
  auto metadata = Metadata();
  if (!metadata.has_value()) {
    LOG(ERROR)
        << "Failed to get DataType due to error metadata: "
        << value_.get<TensorHandle>().GetAsyncMetadata().GetError().message();
    return tensorflow::DT_INVALID;
  }
  auto kind = metadata.value()->dtype;
  if (kind == DType::Unsupported) {
    AsyncValue* async_tensor = value_.get<TensorHandle>().GetAsyncTensor();
    if (!async_tensor->IsAvailable()) {
      context_.GetHostContext()->Await(FormRef(async_tensor));
    }

    if (async_tensor->IsError()) {
      LOG(ERROR) << "Failed to get DataType from an error tensor "
                 << async_tensor->GetError().message();
      return tensorflow::DT_INVALID;
    }
    assert(async_tensor->IsType<tensorflow::tfd::RuntimeFallbackTensor>());
    return async_tensor->get<tensorflow::tfd::RuntimeFallbackTensor>()
        .GetTensorHandle()
        ->DataType();
  }
  return ConvertDType(kind);
}

tensorflow::Status TensorHandleInterface::TensorHandleStatus() const {
  if (context_.IsAsync()) {
    return ::tensorflow::OkStatus();
  } else {
    auto metadata = Metadata();
    if (!metadata.has_value()) {
      LOG(ERROR)
          << "Metadata in the tensor handle is an error metadata: "
          << value_.get<TensorHandle>().GetAsyncMetadata().GetError().message();
      return tensorflow::errors::Internal(
          value_.get<TensorHandle>().GetAsyncMetadata().GetError().message());
    }

    AsyncValue* async_tensor = value_.get<TensorHandle>().GetAsyncTensor();
    if (!async_tensor->IsAvailable()) {
      context_.GetHostContext()->Await(FormRef(async_tensor));
    }

    if (async_tensor->IsError()) {
      LOG(ERROR) << "Async tensor in the tensor handle is an error tensor: "
                 << async_tensor->GetError().message();
      return tensorflow::errors::Internal(async_tensor->GetError().message());
    }

    return ::tensorflow::OkStatus();
  }
}

tensorflow::Status TensorHandleInterface::Shape(
    tensorflow::PartialTensorShape* shape) const {
  auto metadata = Metadata();
  if (!metadata.has_value()) {
    return tensorflow::FromAbslStatus(
        value_.get<TensorHandle>().GetAsyncMetadata().GetError());
  }
  int num_dims = metadata.value()->shape.GetRank();
  if (num_dims == -1) {
    return ::tensorflow::OkStatus();
  }
  llvm::SmallVector<Index, 8> dims;
  metadata.value()->shape.GetDimensions(&dims);
  TF_RETURN_IF_ERROR(tensorflow::TensorShapeUtils::MakeShape(dims, shape));
  return ::tensorflow::OkStatus();
}

tensorflow::Status TensorHandleInterface::NumDims(int* num_dims) const {
  auto metadata = Metadata();
  if (!metadata.has_value()) {
    return tensorflow::FromAbslStatus(
        value_.get<TensorHandle>().GetAsyncMetadata().GetError());
  }
  *num_dims = metadata.value()->shape.GetRank();

  return ::tensorflow::OkStatus();
}

tensorflow::Status TensorHandleInterface::NumElements(
    int64_t* num_elements) const {
  auto metadata = Metadata();
  if (!metadata.has_value()) {
    return tensorflow::FromAbslStatus(
        value_.get<TensorHandle>().GetAsyncMetadata().GetError());
  }
  *num_elements = metadata.value()->shape.GetNumElements();

  return ::tensorflow::OkStatus();
}

tensorflow::Status TensorHandleInterface::Dim(int dim_index,
                                              int64_t* dim) const {
  auto metadata = Metadata();
  if (!metadata.has_value()) {
    return tensorflow::FromAbslStatus(
        value_.get<TensorHandle>().GetAsyncMetadata().GetError());
  }
  *dim = metadata.value()->shape.GetDimensionSize(dim_index);

  return ::tensorflow::OkStatus();
}

const char* TensorHandleInterface::DeviceName(
    tensorflow::Status* status) const {
  auto& th = value_.get<TensorHandle>();
  if (!th.IsDeviceAvailable()) {
    context_.GetHostContext()->Await(th.GetAsyncDevice().CopyRCRef());
  }
  if (th.IsDeviceError()) {
    *status = tensorflow::FromAbslStatus(th.GetAsyncDevice().GetError());
    return nullptr;
  }
  return th.GetAvailableDevice()->name().data();
}

const char* TensorHandleInterface::BackingDeviceName(
    tensorflow::Status* status) const {
  return DeviceName(status);
}

const char* TensorHandleInterface::DeviceType(
    tensorflow::Status* status) const {
  auto& th = value_.get<TensorHandle>();
  if (!th.IsDeviceAvailable()) {
    context_.GetHostContext()->Await(th.GetAsyncDevice().CopyRCRef());
  }
  if (th.IsDeviceError()) {
    *status = tensorflow::FromAbslStatus(th.GetAsyncDevice().GetError());
    return nullptr;
  }
  return th.GetAvailableDevice()->type().name().data();
}

tensorflow::AbstractTensorInterface* TensorHandleInterface::Resolve(
    tensorflow::Status* status) {
  auto* host_ctx = context_.GetHostContext();
  auto host_device_ref = host_ctx->GetHostDeviceRef();
  auto& th = value_.get<TensorHandle>();

  auto tensor_av = th.GetAsyncTensor();
  if (!tensor_av->IsAvailable()) {
    host_ctx->Await(FormRef(tensor_av));
  }
  if (auto* error = tensor_av->GetErrorIfPresent()) {
    *status = tensorflow::FromAbslStatus(*error);
    return nullptr;
  }
  assert(th.IsMetadataAvailable());

  if (th.GetAsyncTensor()->get<Tensor>().tensor_type() ==
      StringHostTensor::kTensorType) {
    tensorflow::Tensor tf_tensor =
        tensorflow::tfd::CopyShtToTfTensor(tensor_av->get<StringHostTensor>());
    return new tensorflow::TensorInterface(tf_tensor);
  }

  // Convert the tensor to DenseHostTensor.
  auto req_ctx =
      tfrt::RequestContextBuilder(host_ctx, context_.GetResourceContext())
          .build();
  if (!req_ctx) {
    *status = tensorflow::Status(
        tensorflow::error::Code::UNKNOWN,
        StrCat("Failed to build a RequestContext: ", req_ctx.takeError()));
    return nullptr;
  }
  tfrt::ExecutionContext exec_ctx{std::move(*req_ctx)};
  auto target_th = th.TransferTo(exec_ctx, std::move(host_device_ref),
                                 DenseHostTensor::kTensorType);

  auto target_av = target_th.GetAsyncTensor();
  if (!target_av->IsAvailable()) {
    host_ctx->Await(FormRef(target_av));
  }
  if (target_av->IsError()) {
    *status = tensorflow::Status(
        tensorflow::error::Code::UNKNOWN,
        StrCat("Cannot resolve tensor: ", target_av->GetError().message()));
    return nullptr;
  }
  auto host_tensor_ref = target_th.ReleaseTensorRef();
  return new TensorInterface(std::move(host_tensor_ref));
}

llvm::Optional<const TensorMetadata*> TensorHandleInterface::Metadata() const {
  auto& th = value_.get<TensorHandle>();
  if (!th.IsMetadataAvailable()) {
    context_.GetHostContext()->Await(th.GetAsyncMetadata().CopyRCRef());
  }
  if (th.IsMetadataError()) {
    return std::nullopt;
  }
  return &th.GetAvailableMetadata();
}

ContextInterface::ContextInterface(
    const tensorflow::SessionOptions& opts,
    tensorflow::ContextDevicePlacementPolicy default_device_placement_policy,
    bool is_async)
    : ImmediateExecutionContext(kTfrt),
      context_(opts, default_device_placement_policy, is_async) {
  LOG(INFO) << "TFRT Enabled";
  metrics::AddTFRTVersionMetric();

  op_handler_selector_ = std::make_unique<EagerOpHandlerSelector>(
      GetCoreRuntime(), GetEagerContext(), GetFallbackOpHandler(),
      GetEagerContext()->PinSmallOpsToCPU());

  run_metadata_ = std::make_unique<tensorflow::RunMetadata>();
}

ContextInterface::~ContextInterface() {}

AsyncValueRef<Chain>* ContextInterface::GetChain() {
  auto thread_id = std::this_thread::get_id();
  {
    tensorflow::tf_shared_lock l(chain_map_mu_);
    auto it = thread_local_chain_.find(thread_id);
    if (it != thread_local_chain_.end()) {
      return &it->second;
    }
  }
  {
    tensorflow::mutex_lock l(chain_map_mu_);
    if (thread_local_chain_.find(thread_id) == thread_local_chain_.end()) {
      auto chain = GetReadyChain();
      thread_local_chain_[thread_id] = std::move(chain);
    }
    return &thread_local_chain_[thread_id];
  }
}

template <typename T>
static TensorInterface* MakeScalarTensor(T value, HostContext* host) {
  // The TensorInterface implementation assumes the tensor is a DenseHostTensor,
  // so we need to use a DenseHostTensor to represent a scalar tensor.
  TensorMetadata md(GetDType<T>(), {});
  auto t = DenseHostTensor::CreateUninitialized(md, host);
  if (!t) {
    LOG(ERROR) << "Failed to create DenseHostTensor";
    return nullptr;
  }
  auto& dht = t.value();
  MutableDHTArrayView<T> view{&dht};
  view.Elements()[0] = value;

  return new TensorInterface(
      MakeAvailableAsyncValueRef<DenseHostTensor>(std::move(dht)));
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateInt64Scalar(
    int64_t value) {
  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateUint64Scalar(
    uint64_t value) {
  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateInt32Scalar(
    int32_t value) {
  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateFloatScalar(
    float value) {
  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateDoubleScalar(
    double value) {
  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateHalfScalar(
    Eigen::half value) {
  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateStringScalar(
    tensorflow::tstring value) {
  auto* host = GetHostContext();
  TensorMetadata md(DType(DType::String), {});
  auto t = StringHostTensor::MakeConstructedAsyncValueRef(md, host);
  if (t.IsError()) {
    LOG(ERROR) << "Failed to create StringHostTensor";
    return nullptr;
  }
  t->strings()[0] = value;

  t.SetStateConcrete();
  return new TensorInterface(std::move(t));
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateComplex128Scalar(
    tensorflow::complex128 value) {
  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateBoolScalar(
    bool value) {
  return MakeScalarTensor(value, GetHostContext());
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateTensor(
    tensorflow::DataType dtype, absl::Span<const int64_t> dim_sizes) {
  std::vector<Index> dimvec(dim_sizes.size());
  for (int i = 0; i < dim_sizes.size(); ++i) {
    dimvec[i] = static_cast<int64_t>(dim_sizes[i]);
  }

  TensorMetadata md;
  switch (dtype) {
    case tensorflow::DT_UINT8:
      md = CreateMetadata(DType::UI8, dimvec);
      break;
    case tensorflow::DT_INT8:
      md = CreateMetadata(DType::I8, dimvec);
      break;
    case tensorflow::DT_INT16:
      md = CreateMetadata(DType::I16, dimvec);
      break;
    case tensorflow::DT_INT32:
      md = CreateMetadata(DType::I32, dimvec);
      break;
    case tensorflow::DT_INT64:
      md = CreateMetadata(DType::I64, dimvec);
      break;
    case tensorflow::DT_HALF:
      md = CreateMetadata(DType::F16, dimvec);
      break;
    case tensorflow::DT_FLOAT:
      md = CreateMetadata(DType::F32, dimvec);
      break;
    case tensorflow::DT_DOUBLE:
      md = CreateMetadata(DType::F64, dimvec);
      break;
    case tensorflow::DT_BOOL:
      md = CreateMetadata(DType::I1, dimvec);
      break;
    case tensorflow::DT_COMPLEX64:
      md = CreateMetadata(DType::Complex64, dimvec);
      break;
    case tensorflow::DT_COMPLEX128:
      md = CreateMetadata(DType::Complex128, dimvec);
      break;
    case tensorflow::DT_VARIANT:
      // Note: TF Python API can create variant tensor for ragged tensor.
      md = CreateMetadata(DType::Variant, dimvec);
      break;
    case tensorflow::DT_STRING:
      // No TFRT Metadata needed for non-scalar string tensors.
      break;
    default:
      LOG(ERROR) << "Cannot create tensor with dtype: " << dtype;
      return nullptr;
  }

  if (dtype == tensorflow::DT_STRING) {
    // Create Tensorflow Tensor as a buffer for tstrings.
    return new TensorInterface(
        tensorflow::Tensor(dtype, tensorflow::TensorShape(dim_sizes)));
  } else {
    auto t = DenseHostTensor::CreateUninitialized(md, GetHostContext());
    return new TensorInterface(
        MakeAvailableAsyncValueRef<DenseHostTensor>(std::move(t.value())));
  }
}

tensorflow::AbstractTensorInterface* ContextInterface::CreateTensor(
    tensorflow::DataType dtype, const int64_t* dims, int num_dims, void* data,
    size_t len, MemoryReleaser memory_releaser, void* memory_releaser_arg) {
  TensorMetadata metadata(ConvertDType(dtype),
                          {dims, static_cast<size_t>(num_dims)});
  RCReference<HostBuffer> buffer = HostBuffer::CreateFromExternal(
      data, len,
      [memory_releaser, memory_releaser_arg](void* data, size_t len) {
        memory_releaser(data, len, memory_releaser_arg);
      });
  AsyncValueRef<DenseHostTensor> dht =
      MakeConstructedAsyncValueRef<DenseHostTensor>(metadata,
                                                    std::move(buffer));

  dht.SetStateConcrete();
  return new TensorInterface(std::move(dht));
}

bool ContextInterface::UsesTFRT() { return true; }

tensorflow::ImmediateExecutionTensorHandle* ContextInterface::CreateLocalHandle(
    tensorflow::AbstractTensorInterface* t) {
  auto* tensor_interface = down_cast<TensorInterface*>(t);
  auto* host = GetHostContext();

  // Create RuntimeFallbackTensor from a TF Tensor, and then create
  // the according TensorHandleInterface.
  if (tensor_interface->IsTfTensor()) {
    tensorflow::tfd::OwnedTensorHandle tf_tensor_handle{
        tensorflow::TensorHandle::CreateLocalHandle(
            tensor_interface->TfTensor())};

    auto expected_result_tensor =
        tensorflow::tfd::CreateRuntimeFallbackTensorFromTfTensorHandle(
            std::move(tf_tensor_handle), GetHostContext());

    if (expected_result_tensor) {
      return new TensorHandleInterface(
          Value(TensorHandle(host->GetHostDeviceRef(),
                             expected_result_tensor.get().metadata(),
                             MakeAvailableAsyncValueRef<
                                 tensorflow::tfd::RuntimeFallbackTensor>(
                                 std::move(expected_result_tensor.get())))),
          GetTfrtContext());
    } else {
      return new TensorHandleInterface(
          Value(TensorHandle::CreateError(MakeErrorAsyncValueRef(
              StrCat(expected_result_tensor.takeError())))),
          GetTfrtContext());
    }
  }

  auto tensor_av = tensor_interface->TensorRef();
  const TensorMetadata& md = tensor_av.get<Tensor>().metadata();

  // NOTE(fishx): Following logic is needed to let TF-TFRT fully reach
  // performance parity with current TF. This API is used to by tf.constant
  // to convert Python object to **CPU** Tensor. tf.constant in current TF
  // heavily depends on Tensor Mirroring feature for good performance. However,
  // TFRT does not have Tensor Mirroring feature. In order to use Tensor
  // Mirroring from current TF runtime, we convert the result of tf.constant to
  // Fallback Tensor.

  if (tensor_av.IsAvailable()) {
    if (auto* dht = llvm::dyn_cast<DenseHostTensor>(&tensor_av.get<Tensor>())) {
      return new TensorHandleInterface(
          Value(TensorHandle(
              host->GetHostDeviceRef(), md,
              MakeAvailableAsyncValueRef<
                  tensorflow::tfd::RuntimeFallbackTensor>(
                  tensorflow::tfd::CopyRefDHTToRuntimeFallbackTensor(*dht,
                                                                     host)))),
          GetTfrtContext());
    }
  } else {
    auto result_tensor = MakeIndirectAsyncValue();
    tensor_av.AndThen([host, result_tensor = result_tensor,
                       tensor_av = tensor_av.CopyRef()]() {
      if (auto* dht =
              llvm::dyn_cast<DenseHostTensor>(&tensor_av.get<Tensor>())) {
        result_tensor->ForwardTo(
            MakeAvailableAsyncValueRef<tensorflow::tfd::RuntimeFallbackTensor>(
                tensorflow::tfd::CopyRefDHTToRuntimeFallbackTensor(*dht,
                                                                   host)));
      } else {
        result_tensor->ForwardTo(tensor_av.CopyRef());
      }
    });
    return new TensorHandleInterface(
        Value(TensorHandle(host->GetHostDeviceRef(), md,
                           AsyncValueRef<Tensor>(std::move(result_tensor)))),
        GetTfrtContext());
  }
  return new TensorHandleInterface(
      Value(TensorHandle(host->GetHostDeviceRef(), md, std::move(tensor_av))),
      GetTfrtContext());
}

tensorflow::ImmediateExecutionTensorHandle*
ContextInterface::CreateLocalHandleFromTFTensor(tensorflow::Tensor& t,
                                                const char* d_name) {
  auto* host = GetHostContext();
  // Create RuntimeFallbackTensor from a TF Tensor, and then create
  // the according TensorHandleInterface.
  tensorflow::tfd::OwnedTensorHandle tf_tensor_handle{
      tensorflow::TensorHandle::CreateLocalHandle(std::move(t))};

  tfrt::Expected<tensorflow::tfd::RuntimeFallbackTensor>
      expected_result_tensor =
          tensorflow::tfd::CreateRuntimeFallbackTensorFromTfTensorHandle(
              std::move(tf_tensor_handle), GetHostContext());

  if (expected_result_tensor) {
    return new TensorHandleInterface(
        Value(TensorHandle(
            host->GetHostDeviceRef(), expected_result_tensor.get().metadata(),
            MakeAvailableAsyncValueRef<tensorflow::tfd::RuntimeFallbackTensor>(
                std::move(expected_result_tensor.get())))),
        GetTfrtContext());
  } else {
    return new TensorHandleInterface(
        Value(TensorHandle::CreateError(MakeErrorAsyncValueRef(
            StrCat(expected_result_tensor.takeError())))),
        GetTfrtContext());
  }
}

tensorflow::ImmediateExecutionTensorHandle*
ContextInterface::TFTensorHandleFromInterface(
    tensorflow::ImmediateExecutionTensorHandle* handle) {
  TensorHandle th = tfrt::tf::TensorHandleFromInterface(handle)->Handle();
  AsyncValue* tensor_av = th.GetAsyncTensor();
  if (tensor_av->IsUnavailable()) GetHostContext()->Await(FormRef(tensor_av));

  auto& tensor = th.GetAsyncTensor()->get<Tensor>();

  if (auto* rtfbt =
          llvm::dyn_cast<tensorflow::tfd::RuntimeFallbackTensor>(&tensor))
    return rtfbt->GetTensorHandle();

  if (auto* dht = llvm::dyn_cast<tfrt::DenseHostTensor>(&tensor)) {
    return tensorflow::TensorHandle::CreateLocalHandle(
        tensorflow::tfd::MoveHostBufferToTfTensor(dht->buffer(), dht->dtype(),
                                                  dht->shape()));
  }

  if (auto* sht = llvm::dyn_cast<tfrt::StringHostTensor>(&tensor)) {
    return tensorflow::TensorHandle::CreateLocalHandle(
        tensorflow::tfd::CopyShtToTfTensor(*sht));
  }

  LOG(ERROR) << "Unsupported tensor type";
  return nullptr;
}

tensorflow::ImmediateExecutionOperation* ContextInterface::CreateOperation() {
  return new OperationInterface(this);
}

// TODO(srbs): Change this to directly fetch the MLIR function once that is
// supported.
tensorflow::Status ContextInterface::RegisterFunction(
    tensorflow::AbstractFunction* f) {
  tensorflow::FunctionDef* fdef;
  TF_RETURN_IF_ERROR(f->GetFunctionDef(&fdef));
  if (!fdef) {
    return tensorflow::errors::InvalidArgument(
        "GetFunctionDef returned nullptr.");
  }
  return AddFunctionDef(*fdef);
}

void ContextInterface::ListDevices(
    std::vector<tensorflow::DeviceAttributes>* devices) {
  context_.GetEagerContext()->ListDevices(devices);
}

tensorflow::Status ContextInterface::AddDevices(
    std::vector<std::unique_ptr<tensorflow::Device>> devices) {
  if (!devices.empty() && devices[0]->device_type() != "CPU")
    return tensorflow::errors::InvalidArgument(
        "Device: ", devices[0]->device_type(), " is not allowed to be added ",
        "after the context is initialized. Currently allowed device: CPU. ",
        "May update this API to allow adding more types of devices.");

  for (const auto& d : devices) {
    GetHostContext()->GetDeviceManager()->MaybeAddDevice(
        TakeRef(new CpuDevice(d->name())));
  }
  TF_RETURN_IF_ERROR(GetEagerContext()->AddDevices(std::move(devices)));

  return ::tensorflow::OkStatus();
}

void ContextInterface::ClearCachesAndThreadExecutors() {
  GetEagerContext()->ClearCachesAndThreadExecutors();
  GetHostContext()->Quiesce();
}

void ContextInterface::StartStep() { GetEagerContext()->StartStep(); }

void ContextInterface::EndStep() { GetEagerContext()->EndStep(); }

tensorflow::Status ContextInterface::EnableCollectiveOps(
    const tensorflow::ServerDef& server_def) {
  // Preserve the local virtual device names, since local virtual devices are
  // added by TFRT and we need to add it back after worker server is
  // initialized. Currently one such use case is the TPU_SYSTEM device, which
  // is a virtual device specifically used to initialize TPUs.
  std::vector<std::string> virtual_device_names;
  int64_t ncpus = 0;

  for (const auto& d :
       GetHostContext()->GetDeviceManager()->ListDevices<Device>()) {
    if (d->IsDeviceType(tfrt::VirtualDevice::kDeviceType)) {
      tensorflow::DeviceNameUtils::ParsedName p;
      if (!tensorflow::DeviceNameUtils::ParseFullName(d->name().str(), &p)) {
        return tensorflow::errors::InvalidArgument(
            "Invalid local virtual device name: ", d->name().str());
      }

      virtual_device_names.push_back(tensorflow::DeviceNameUtils::FullName(
          server_def.job_name(), /*replica=*/0, server_def.task_index(), p.type,
          p.id));
    }
    if (d->IsDeviceType(tfrt::CpuDevice::kDeviceType)) {
      ++ncpus;
    }
  }

  TF_RETURN_IF_ERROR(GetEagerContext()->EnableCollectiveOps(server_def));

  // Create new devices with updated device name.
  std::vector<std::unique_ptr<tensorflow::Device>> dummy_tf_devices;
  CreateDummyTfDevices(virtual_device_names, &dummy_tf_devices);

  std::string name_prefix =
      absl::StrCat("/job:", server_def.job_name(),
                   "/replica:0/task:", server_def.task_index());

  // Update host device in TFRT HostContext.
  GetHostContext()->ResetHostDevice(
      GetHostContext()
          ->GetDeviceManager()
          ->MaybeAddDevice(
              MakeRef<CpuDevice>(absl::StrCat(name_prefix, "/device:CPU:0")))
          .release());

  // Create additional host logical CPU devices.
  for (int64_t i = 1; i < ncpus; ++i) {
    GetHostContext()->GetDeviceManager()->MaybeAddDevice(
        MakeRef<CpuDevice>(absl::StrCat(name_prefix, "/device:CPU:", i)));
  }
  // Update virtual devices in TFRT HostContext.
  AddDummyTfrtDevices(virtual_device_names, GetHostContext());

  // Update eager context's device manager.
  auto* local_device_mgr = dynamic_cast<tensorflow::DynamicDeviceMgr*>(
      GetEagerContext()->local_device_mgr());
  TF_RETURN_IF_ERROR(local_device_mgr->AddDevices(std::move(dummy_tf_devices)));

  return ::tensorflow::OkStatus();
}

tensorflow::Status ContextInterface::BuildFunctionRequestContext(
    tensorflow::tfrt_stub::OpKernelRunnerTable* runner_table,
    RCReference<tfrt::RequestContext>* request_context) {
  auto* step_container = GetEagerContext()->StepContainer();
  RequestContextBuilder request_context_builder(
      GetHostContext(), GetResourceContext(), step_container->StepId());

  TF_RETURN_IF_ERROR(tensorflow::tfd::SetUpKernelFallbackCompatRequestContext(
      &request_context_builder, runner_table, GetEagerContext()));
  auto expected_request_context = std::move(request_context_builder).build();
  if (!expected_request_context) {
    return tensorflow::errors::Internal(
        StrCat(expected_request_context.takeError()));
  }
  *request_context = std::move(expected_request_context.get());
  return ::tensorflow::OkStatus();
}

tensorflow::Status ContextInterface::BuildOpRequestContext(
    RCReference<tfrt::RequestContext>* request_context) {
  return BuildFunctionRequestContext(/*runner_table=*/nullptr, request_context);
}

tensorflow::ImmediateExecutionTensorHandle*
ContextInterface::CopyTensorHandleToDevice(
    tensorflow::ImmediateExecutionTensorHandle* handle, const char* device_name,
    tensorflow::Status* status) {
  auto* host_ctx = GetHostContext();

  TensorHandle src_th = tfrt::tf::TensorHandleFromInterface(handle)->Handle();

  auto tfrt_device_name =
      ConvertTfDeviceNameToTfrt(device_name, GetEagerContext());
  if (!tfrt_device_name) {
    *status = tensorflow::errors::InvalidArgument(
        StrCat(tfrt_device_name.takeError()));
    RCReference<AsyncValue> error_av =
        MakeErrorAsyncValueRef(status->error_message());
    return new TensorHandleInterface(
        Value(TensorHandle::CreateError(std::move(error_av))),
        GetTfrtContext());
  }
  auto dst_device_ref = host_ctx->GetDeviceManager()->GetDeviceRef<Device>(
      tfrt_device_name.get());
  if (!dst_device_ref) {
    std::string error_message =
        tfrt::StrCat("Failed to find destination device with name: ",
                     tfrt_device_name.get());
    *status = tensorflow::errors::Internal(error_message);
    RCReference<AsyncValue> error_av = MakeErrorAsyncValueRef(error_message);
    return new TensorHandleInterface(
        Value(TensorHandle::CreateError(std::move(error_av))),
        GetTfrtContext());
  }

  RCReference<RequestContext> request_ctx;
  *status = BuildOpRequestContext(&request_ctx);
  if (!status->ok()) return nullptr;

  ExecutionContext exec_ctx{std::move(request_ctx)};

  auto target_th =
      src_th.TransferToInferredType(exec_ctx, std::move(dst_device_ref));

  auto target_av = target_th.GetAsyncTensor();
  if (target_av->IsError()) {
    *status = tensorflow::errors::Internal(
        tfrt::StrCat("Copying to device <", tfrt_device_name.get(),
                     "> failed: ", target_av->GetError().message()));
    return nullptr;
  }
  return new TensorHandleInterface(Value(target_th.CopyRef()),
                                   GetTfrtContext());
}

tensorflow::Status ContextInterface::AddFunctionDef(
    const tensorflow::FunctionDef& fdef) {
  return GetEagerContext()->AddFunctionDef(fdef);
}

tensorflow::Status ContextInterface::AddFunctionDefWithStackTraces(
    const tensorflow::FunctionDef& fdef,
    const tensorflow::StackTracesMap& stack_traces) {
  return GetEagerContext()->AddFunctionDefWithStackTraces(fdef, stack_traces);
}

std::vector<std::string> ContextInterface::ListFunctionNames() {
  return GetEagerContext()->ListFunctionNames();
}

tensorflow::Status ContextInterface::RemoveFunction(const std::string& func) {
  // TODO(tfrt-devs): We need to ensure all invocations of this function is
  // finished before removing it.
  function_cache_.RemoveFunction(func);
  return GetEagerContext()->RemoveFunction(func);
}

const tensorflow::FunctionDef* ContextInterface::FindFunctionDef(
    const std::string& name) const {
  return GetEagerContext()->FindFunctionDef(name);
}

const tensorflow::DeviceNameUtils::ParsedName&
ContextInterface::HostCPUParsedName() const {
  return context_.HostCPUParsedName();
}

const std::string& ContextInterface::HostCPUName() const {
  return context_.GetEagerContext()->HostCPUName();
}

tensorflow::CustomDeviceOpHandler&
ContextInterface::GetCustomDeviceOpHandler() {
  return context_.GetEagerContext()->GetCustomDeviceOpHandler();
}

bool ContextInterface::IsCustomDevice(const std::string& device_name) {
  return context_.GetEagerContext()->IsCustomDevice(device_name);
}

tensorflow::Status ContextInterface::RegisterCustomDevice(
    const std::string& name, std::unique_ptr<tensorflow::CustomDevice> device) {
  return context_.GetEagerContext()->RegisterCustomDevice(name,
                                                          std::move(device));
}

tensorflow::FunctionLibraryDefinition* ContextInterface::FuncLibDef() {
  return context_.GetEagerContext()->FuncLibDef();
}

void ContextInterface::SetReuseRendezvousForFunctions(
    bool reuse_rendezvous_for_functions) {
  // TODO(fishx): This feature doesn't work properly in TFRT yet. Fix it.
  context_.GetEagerContext()->SetReuseRendezvousForFunctions(
      reuse_rendezvous_for_functions);
}

void ContextInterface::ResetGlobalRendezvousForFunction() {
  context_.GetEagerContext()->ResetGlobalRendezvousForFunction();
}

std::vector<std::string> ContextInterface::GetLoggedOpsTestonly() {
  const auto& ret = GetHostContext()
                        ->GetOrCreateSharedContext<tensorflow::tfd::OpLogger>()
                        .GetLoggedOps();
  return std::vector<std::string>(ret.begin(), ret.end());
}

HostContext* ContextInterface::GetHostContext() {
  return GetCoreRuntime()->GetHostContext();
}

tensorflow::EagerContext* ContextInterface::GetEagerContext() {
  return context_.GetEagerContext();
}

const tensorflow::EagerContext* ContextInterface::GetEagerContext() const {
  return context_.GetEagerContext();
}

CoreRuntime* ContextInterface::GetCoreRuntime() {
  return context_.GetCoreRuntime();
}

TfrtContext* ContextInterface::GetTfrtContext() { return &context_; }

OpHandler* ContextInterface::GetFallbackOpHandler() {
  return context_.GetFallbackOpHandler();
}

ResourceContext* ContextInterface::GetResourceContext() {
  return context_.GetResourceContext();
}

tensorflow::Status ContextInterface::SelectOpHandlerFromArguments(
    const tensorflow::ImmediateExecutionOperation& op, OpHandler** op_handler) {
  return op_handler_selector_->SelectFromArguments(op, op_handler);
}

tensorflow::Status ContextInterface::SelectOpHandlerFromNodeDef(
    const tensorflow::ImmediateExecutionOperation& op, const NodeDef* node_def,
    OpHandler** op_handler) {
  return op_handler_selector_->SelectFromNodeDef(op, node_def, op_handler);
}

std::unique_ptr<tensorflow::RunMetadata> ContextInterface::ExportRunMetadata() {
  mutex_lock l(run_metadata_mu_);

  // NOTE(fishx): We need to merge run_metadata from TF Eager Context because
  // right now we still use current TF runtime to execute graph (e.g. tf.data
  // via fallback).
  auto result = GetEagerContext()->ExportRunMetadata();
  result->MergeFrom(*run_metadata_);
  run_metadata_ = std::make_unique<tensorflow::RunMetadata>();

  return result;
}

tensorflow::Status ContextInterface::RunMetadataRecordFunction(
    const std::string& func_name) {
  const tensorflow::FunctionDef* fdef =
      GetEagerContext()->FindFunctionDef(func_name);
  if (fdef == nullptr) {
    return tensorflow::errors::InvalidArgument(
        "Failed to find function \"", func_name, "\" in function library");
  }
  std::unique_ptr<tensorflow::FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *fdef, tensorflow::AttrSlice(), GetEagerContext()->FuncLibDef(), &fbody));
  tensorflow::GraphDef def;
  fbody->graph->ToGraphDef(&def);
  *def.mutable_library() =
      GetEagerContext()->FuncLibDef()->ReachableDefinitions(def).ToProto();

  mutex_lock l(run_metadata_mu_);
  auto* function_graphs = run_metadata_->add_function_graphs();
  *function_graphs->mutable_pre_optimization_graph() = def;
  // TODO(b/171600738): Figure out a way to record the right post optimization
  // graph and partition graph.
  *function_graphs->mutable_post_optimization_graph() = def;
  *function_graphs->add_partition_graphs() = def;
  *run_metadata_->add_partition_graphs() = def;
  return ::tensorflow::OkStatus();
}

void ContextInterface::SetExecutorForThread(
    tensorflow::EagerExecutor* executor) {
  GetEagerContext()->SetExecutorForThread(executor);
}

tfrt::Location AbortLocationHandler::GetCurrentLocation() {
  return tfrt::Location(this, GetNextLocationId());
}

void OpAttrsInterface::GetNameAttrList(
    tensorflow::NameAttrList* name_and_attrs) const {
  fallback_attrs_->FillAttrValueMap(name_and_attrs->mutable_attr());
  name_and_attrs->set_name(fallback_attrs_->op_name());
}

Status OpAttrsInterface::GetTypeList(
    absl::string_view attr_name,
    absl::InlinedVector<tensorflow::DataType, 4>* type_list) const {
  return tensorflow::errors::Unimplemented("OpAttrsInterface::GetTypeList");
}

bool OpAttrsInterface::GetInt(absl::string_view attr_name,
                              int64_t* result) const {
  return attrs_->Get<int64_t>({attr_name.data(), attr_name.size()}, result);
}

bool OpAttrsInterface::GetFloat(absl::string_view attr_name,
                                float* result) const {
  return attrs_->Get<float>({attr_name.data(), attr_name.size()}, result);
}

bool OpAttrsInterface::GetBool(absl::string_view attr_name,
                               bool* result) const {
  return attrs_->Get<bool>({attr_name.data(), attr_name.size()}, result);
}

bool OpAttrsInterface::GetType(absl::string_view attr_name,
                               tensorflow::DataType* result) const {
  auto optional_type =
      attrs_->GetOptional<OpAttrType>({attr_name.data(), attr_name.size()});
  if (!optional_type.has_value()) return false;
  *result = tensorflow::tfd::ConvertToTfDataType(optional_type.value());
  return true;
}

OperationInterface::OperationInterface(ContextInterface* context)
    : ImmediateExecutionOperation(kTfrt),
      op_attrs_(&attrs_, &fallback_attrs_),
      context_(context) {}

tensorflow::Status OperationInterface::Reset(const char* op,
                                             const char* raw_device_name) {
  op_name_ = op;
  args_.clear();
  attrs_.Reset();
  custom_device_tensor_handle_count_ = 0;
  op_def_ = nullptr;
  fallback_attrs_.Reset(op);
  stack_trace_.reset();
  op_ = nullptr;
  function_state_.reset();
  tensorflow::Status s = tensorflow::OpDefForOp(op_name_, &op_def_);
  is_function_ = !s.ok();
  return SetDeviceName(raw_device_name);
}

tensorflow::Status OperationInterface::Execute(
    absl::Span<tensorflow::AbstractTensorHandle*> retvals, int* num_retvals) {
  tensorflow::profiler::TraceMe trace(
      [&] {
        return absl::StrCat("TFRT_Execute:", Name(), " device:", DeviceName());
      },
      tensorflow::profiler::TraceMeLevel::kInfo);
  if (custom_device_tensor_handle_count_ > 0) {
    return tensorflow::errors::InvalidArgument(
        "Cannot execute ops that conntains unsupported arg in TFRT.");
  }

  TF_RETURN_IF_ERROR(Initialize());
  assert(op_ != nullptr || function_state_);
  auto* corert = context_->GetCoreRuntime();
  auto* chain = context_->GetChain();
  auto* host = corert->GetHostContext();
  llvm::SmallVector<TensorHandle, 8> th_args;
  th_args.reserve(args_.size());

  llvm::SmallVector<TensorHandle, 8> result_ths;
  result_ths.resize(*num_retvals);

  if (function_state_) {
    // Set up arguments. Check argument dtype synchronously if available.
    auto arg_types = function_state_->GetArgTypes();
    if (args_.size() != arg_types.size()) {
      return tensorflow::errors::InvalidArgument("Expects ", arg_types.size(),
                                                 " arguments, but ",
                                                 args_.size(), " is provided");
    }
    auto args_size = args_.size();
    for (auto i = 0; i < args_size; ++i) {
      th_args.push_back(down_cast<TensorHandleInterface*>(args_[i].get())
                            ->Handle()
                            .CopyRef());
      // TODO(b/173556766): This dtype check is only needed for corert lowering.
      // In native lowering, compiler should obtain the argument dtype
      // information from FunctionBody directly and lower the op to the native
      // kernel that accepts the specified dtype.
      if (th_args[i].IsMetadataAvailable()) {
        auto arg_dtype = th_args[i].GetAvailableMetadata().dtype;
        if (arg_dtype != arg_types[i]) {
          return tensorflow::errors::InvalidArgument(
              "Expects arg[", i, "] to be ", arg_types[i], " but ", arg_dtype,
              " is provided");
        }
      }
    }

    RCReference<RequestContext> request_ctx;
    TF_RETURN_IF_ERROR(context_->BuildFunctionRequestContext(
        function_state_->GetRunnerTable(), &request_ctx));

    ExecutionContext exec_ctx{std::move(request_ctx),
                              abort_location_handler_.GetCurrentLocation()};

    // Make BEF executor to use TfThreadPoolWorkQueue to dispatch kernels.
    exec_ctx.set_work_queue(
        context_->GetTfrtContext()->GetTfThreadPoolWorkQueue());

    // Execute the function.
    function_state_->GetFunc()(exec_ctx, th_args, OpAttrsRef(attrs_),
                               result_ths, chain);
  } else {
    RCReference<RequestContext> request_ctx;
    TF_RETURN_IF_ERROR(context_->BuildOpRequestContext(&request_ctx));

    ExecutionContext exec_ctx{std::move(request_ctx),
                              abort_location_handler_.GetCurrentLocation()};
    for (auto& arg : args_) {
      th_args.push_back(
          down_cast<TensorHandleInterface*>(arg.get())->Handle().CopyRef());
    }
    // If the CoreRuntimeOp is a native TFRT op, transfer arguments to target
    // device if necessary.
    if (!op_->IsFallback()) {
      // Get the target device of the arguments that we want to implicitly copy
      // to.
      auto dst_device_ref = op_->GetDeviceRef();

      for (auto& th_arg : th_args) {
        th_arg =
            th_arg.TransferTo(exec_ctx, dst_device_ref, op_->GetTensorType());
      }
    }

    (*op_)(exec_ctx, th_args, OpAttrsRef(attrs_), result_ths, chain);
  }

  tensorflow::Status s = ::tensorflow::OkStatus();

  if (TF_PREDICT_FALSE(!this->context_->IsAsync() && !chain->IsAvailable()))
    host->Await({chain->CopyRCRef()});

  if (TF_PREDICT_FALSE(chain->IsError())) {
    s = tensorflow::FromAbslStatus(chain->GetError());
    // TODO(tfrt-devs): Assess if we need a explicit API to clear error.
    *chain = GetReadyChain();
  }

  for (size_t i = 0, e = result_ths.size(); i != e; ++i) {
    auto& th_ref = result_ths[i];
    if (TF_PREDICT_FALSE(!this->context_->IsAsync() &&
                         !th_ref.GetAsyncTensor()->IsAvailable()))
      host->Await(FormRef(th_ref.GetAsyncTensor()));

    // NOTE(fishx): In async mode, we won't report error synchronously even
    // though it is possible in TFRT. This is intended to match behavior in
    // current TF. However, in the future, we may want to update this
    // behavior since synchronous error may improve user experience in async
    // mode.
    if (TF_PREDICT_FALSE(!this->context_->IsAsync() &&
                         th_ref.GetAsyncTensor()->IsError() && s.ok()))
      s = tensorflow::FromAbslStatus(th_ref.GetAsyncTensor()->GetError());

    if (function_state_ && context_->IsAsync()) {
      retvals[i] = new TensorHandleInterface(function_state_->GetRetTypes()[i],
                                             Value(std::move(result_ths[i])),
                                             context_->GetTfrtContext());
    } else {
      retvals[i] = new TensorHandleInterface(Value(std::move(result_ths[i])),
                                             context_->GetTfrtContext());
    }
  }

  return s;
}

tensorflow::Status OperationInterface::Initialize() {
  CoreRuntime* corert = context_->GetCoreRuntime();
  if (!is_function_) {
    // Obtain input arguments' dtype attrs as part of the cache key.
    llvm::SmallVector<string_view, 4> dtypes;
    attrs_.IterateEntries([&](const OpAttrsRawEntry& entry) {
      if (entry.type == OpAttrType::DTYPE && !entry.IsArray())
        dtypes.push_back(
            GetNameString(*static_cast<const OpAttrType*>(entry.GetData())));
    });

    OpHandler* op_handler = nullptr;
    TF_RETURN_IF_ERROR(
        context_->SelectOpHandlerFromArguments(*this, &op_handler));
    Expected<CoreRuntimeOp*> expected_op = context_->GetOpCache().GetOrAddOp(
        op_name_, op_handler, device_name_, dtypes, this);
    if (!expected_op) {
      return tensorflow::errors::InvalidArgument(
          StrCat("Cannot obtain CoreRuntimeOp: ", op_name_,
                 " on device: ", device_name_, expected_op.takeError()));
    }
    op_ = expected_op.get();
    // Update device name since op_handler_selecter may choose an op_handler
    // that's different from what the user specifies.
    device_name_ = op_->DeviceName().str();
    return ::tensorflow::OkStatus();
  }

  bool compile_with_xla = false;
  GetFuncAttr(attrs_, op_name_, *context_->GetEagerContext()->FuncLibDef(),
              tensorflow::kXlaMustCompileAttr, &compile_with_xla);
  // If the function has compile_with_xla==true, we will use RuntimeFallback
  // to execute it, since TFRT does not support xla yet.
  // TODO(tfrt-devs): Native support of compile_with_xla.
  if (compile_with_xla) {
    Expected<CoreRuntimeOp*> expected_op =
        context_->GetOpCache().GetOrAddXlaOp(op_name_, context_);
    if (!expected_op) {
      return tensorflow::errors::NotFound(
          StrCat("Cannot initialize xla function ", op_name_,
                 " on fallback op handler.", expected_op.takeError()));
    }
    op_ = expected_op.get();
    return ::tensorflow::OkStatus();
  }

  // Note(fishx): We need eager context for now because we need
  // FunctionLibraryDefinition to convert FunctionDef to MLIR TF dialect. In
  // the future, when we can generate MLIR from TF Python, we should get rid of
  // this.
  // FunctionDef -> BEF.
  // Look up the cache. Compile BEF and insert to cache if miss.
  tensorflow::DeviceSet dev_set;
  const DeviceMgr* device_mgr = context_->GetEagerContext()->local_device_mgr();
  if (device_mgr == nullptr)
    return tensorflow::errors::NotFound("Cannot find device manager");
  // TODO(tfrt-devs): support remote devices in TFRT.
  for (auto d : device_mgr->ListDevices()) dev_set.AddDevice(d);
  FunctionCache::FunctionCacheResult result;

  tensorflow::TfrtFunctionCompileOptions compile_options;

  // Use the host device if the user does not place the function to a specific
  // device.
  compile_options.default_device =
      device_name_.empty() ? context_->GetEagerContext()->HostCPUName()
                           : device_name_;

  if (fallback_attrs_.NumAttributes() > 0) {
    const auto& ndef = NodeDef();
    // TODO(tfrt-devs): If we are to create more attributes, consider packing
    // them into a proto.
    {
      const auto& it = ndef.attr().find(kEnableGrapplerAttr);
      if (it != ndef.attr().end()) {
        compile_options.enable_grappler = it->second.b();
      }
    }
  }

  llvm::SmallVector<const tfrt::Device*, 4> input_devices;
  input_devices.reserve(args_.size());
  for (auto& arg : args_) {
    auto arg_th = down_cast<TensorHandleInterface*>(arg.get())->Handle();
    if (!arg_th.IsDeviceAvailable()) {
      corert->GetHostContext()->Await(arg_th.GetAsyncDevice().CopyRCRef());
    }
    input_devices.push_back(down_cast<TensorHandleInterface*>(arg.get())
                                ->Handle()
                                .GetAvailableDevice()
                                .get());
  }
  TF_RETURN_IF_ERROR(context_->GetFunctionCache().GetOrAddFunction(
      op_name_, device_name_, dev_set, context_->GetEagerContext(), corert,
      /*request_ctx_fn=*/
      [this](tensorflow::tfrt_stub::OpKernelRunnerTable* runner_table,
             RCReference<RequestContext>* request_ctx) {
        return context_->BuildFunctionRequestContext(runner_table, request_ctx);
      },
      abort_location_handler_.GetCurrentLocation(), compile_options,
      input_devices, &result));
  // TODO(tfrt-devs): Avoid calling EagerContext::ShouldStoreGraphs().
  if (result.is_cache_miss &&
      context_->GetEagerContext()->ShouldStoreGraphs()) {
    TF_RETURN_IF_ERROR(context_->RunMetadataRecordFunction(op_name_));
  }
  function_state_ = std::move(result.function_state);
  return ::tensorflow::OkStatus();
}

tensorflow::Status OperationInterface::SetDeviceName(const char* name) {
  if (op_ && name != device_name_) {
    return tensorflow::errors::Internal(
        "Failed to update device name. Right now TFRT cannot update device "
        "name of a fallback op if it is initialized.");
  }
  device_name_ = name ? name : "";
  return ::tensorflow::OkStatus();
}

tensorflow::Status OperationInterface::AddInput(
    tensorflow::AbstractTensorHandle* input) {
  tensorflow::ImmediateExecutionTensorHandle* h =
      down_cast<tensorflow::ImmediateExecutionTensorHandle*>(input);
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (tensorflow::CustomDeviceTensorHandle::classof(h)) {
    custom_device_tensor_handle_count_++;
  }
  h->Ref();
  args_.push_back(
      tensorflow::core::RefCountPtr<tensorflow::ImmediateExecutionTensorHandle>(
          h));
  return ::tensorflow::OkStatus();
}

tensorflow::Status OperationInterface::SetInput(
    size_t index, tensorflow::ImmediateExecutionTensorHandle* input) {
  if (index >= args_.size()) {
    return tensorflow::errors::InvalidArgument("Index >= inputs.size: %d >= %d",
                                               index, args_.size());
  }
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (tensorflow::CustomDeviceTensorHandle::classof(args_[index].get())) {
    custom_device_tensor_handle_count_--;
  }
  if (tensorflow::CustomDeviceTensorHandle::classof(input)) {
    custom_device_tensor_handle_count_++;
  }
  input->Ref();
  args_[index] =
      tensorflow::core::RefCountPtr<tensorflow::ImmediateExecutionTensorHandle>(
          input);
  return ::tensorflow::OkStatus();
}

tensorflow::Status OperationInterface::AddInputList(
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs) {
  return tensorflow::errors::Unimplemented(
      "Unimplemented OperationInterface::AddInputList");
}

absl::Span<tensorflow::ImmediateExecutionTensorHandle* const>
OperationInterface::GetInputs() const {
  return absl::MakeSpan(
      reinterpret_cast<tensorflow::ImmediateExecutionTensorHandle* const*>(
          args_.data()),
      args_.size());
}

tensorflow::Status OperationInterface::SetAttrString(const char* attr_name,
                                                     const char* data,
                                                     size_t length) {
  fallback_attrs_.Set(attr_name, tensorflow::StringPiece(data, length));
  if (attrs_.SetString(attr_name, string_view(data, length)))
    return ::tensorflow::OkStatus();
  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrString failed");
}

tensorflow::Status OperationInterface::SetAttrInt(const char* attr_name,
                                                  int64_t value) {
  fallback_attrs_.Set(attr_name, static_cast<int64_t>(value));
  if (attrs_.Set(attr_name, value)) return ::tensorflow::OkStatus();
  return tensorflow::errors::Internal("OperationInterface::SetAttrInt failed");
}

tensorflow::Status OperationInterface::SetAttrFloat(const char* attr_name,
                                                    float value) {
  fallback_attrs_.Set(attr_name, value);
  if (attrs_.Set(attr_name, value)) return ::tensorflow::OkStatus();
  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFloat failed");
}

tensorflow::Status OperationInterface::SetAttrBool(const char* attr_name,
                                                   bool value) {
  fallback_attrs_.Set(attr_name, value);
  if (attrs_.Set(attr_name, value)) return ::tensorflow::OkStatus();
  return tensorflow::errors::Internal("OperationInterface::SetAttrBool failed");
}

tensorflow::Status OperationInterface::SetAttrType(const char* attr_name,
                                                   tensorflow::DataType value) {
  fallback_attrs_.Set(attr_name, value);
  if (value == tensorflow::DT_INVALID) {
    return tensorflow::errors::InvalidArgument(
        "OperationInterface::SetAttrType failed to set DT_INVALID");
  }
  if (attrs_.Set(attr_name,
                 tfrt::GetOpAttrTypeFromDType(
                     tensorflow::tfd::ConvertTfDataTypeToBefAttrType(value))))
    return ::tensorflow::OkStatus();
  // TODO(fishx): Remove this workaround once we support all dtype in TF.
  // This is fine for now since attribute "T", "U", "Tidx" is not used by TFRT
  // native ops.
  if (std::strcmp(attr_name, "T") == 0 || std::strcmp(attr_name, "U") == 0 ||
      std::strcmp(attr_name, "Tidx") == 0) {
    return ::tensorflow::OkStatus();
  }
  return tensorflow::errors::Internal("OperationInterface::SetAttrType failed");
}

tensorflow::Status OperationInterface::SetAttrShape(const char* attr_name,
                                                    const int64_t* dims,
                                                    const int num_dims) {
  // NOTE: This is copied from EagerOperation::SetAttrShape.
  // TODO(b/154554118): Remove the duplication.
  if (num_dims > tensorflow::TensorShape::MaxDimensions()) {
    return tensorflow::errors::InvalidArgument(
        "Value specified for `", attr_name, "` has ", num_dims,
        " dimensions which is over the limit of ",
        tensorflow::TensorShape::MaxDimensions(), ".");
  }

  tensorflow::TensorShapeProto proto;
  size_t offset;
  if (num_dims < 0) {
    proto.set_unknown_rank(true);

    // Set unranked ShapeAttr.
    offset = bef_attr_encoder_.EncodeUnrankedShapeAttr();
  } else {
    for (int d = 0; d < num_dims; ++d) {
      proto.add_dim()->set_size(dims[d]);
    }

    // Set RankedShapeAttr.
    offset =
        bef_attr_encoder_.EncodeRankedShapeAttr(llvm::ArrayRef(dims, num_dims));
  }
  fallback_attrs_.Set(attr_name, proto);

  auto buf = bef_attr_encoder_.TakeResult();
  tfrt::ShapeAttr shape_attr(buf.data() + offset);
  // TODO(tfrt-devs): Avoid the copy.
  if (attrs_.Set(attr_name, shape_attr)) return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrShape failed");
}

tensorflow::Status OperationInterface::SetAttrFunction(
    const char* attr_name, const tensorflow::AbstractOperation* value) {
  auto* value_operation = down_cast<const OperationInterface*>(value);
  // TODO(b/165412867): Set fallback_attrs_ for eager device placement.
  // Consider removing this and rely on TFRT OpAttrs.
  tensorflow::AttrValue attr_value;
  tensorflow::NameAttrList* func = attr_value.mutable_func();
  func->set_name(value->Name());
  fallback_attrs_.Set(attr_name, attr_value);

  if (attrs_.SetFunc(attr_name, {string_view(value_operation->Name())}))
    return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFunction failed");
}

tensorflow::Status OperationInterface::SetAttrFunctionName(
    const char* attr_name, const char* data, size_t length) {
  // TODO(b/165412867): Set fallback_attrs_ for eager device placement.
  // Consider removing this and rely on TFRT OpAttrs.
  tensorflow::AttrValue attr_value;
  tensorflow::NameAttrList* func = attr_value.mutable_func();
  func->set_name(data);
  fallback_attrs_.Set(attr_name, attr_value);

  if (attrs_.SetFunc(attr_name, {data})) return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFunctionName failed");
}

static size_t SerializeTFETensorToDenseAttr(
    tensorflow::AbstractTensorInterface* tensor,
    tfrt::BefAttrEncoder* encoder) {
  std::vector<uint8_t> data;

  const auto element_type =
      tensorflow::tfd::ConvertTfDataTypeToBefAttrType(tensor->Type());
  llvm::SmallVector<int64_t, 4> shape;
  for (int i = 0; i < tensor->NumDims(); ++i) {
    shape.push_back(tensor->Dim(i));
  }
  auto elements = llvm::ArrayRef(
      reinterpret_cast<const uint8_t*>(tensor->Data()), tensor->ByteSize());
  return encoder->EncodeDenseAttr(static_cast<DType>(element_type), shape,
                                  elements);
}

tensorflow::Status OperationInterface::SetAttrTensor(
    const char* attr_name, tensorflow::AbstractTensorInterface* tensor) {
  tfrt::BefAttrEncoder encoder;
  const size_t offset = SerializeTFETensorToDenseAttr(tensor, &encoder);
  auto buffer = encoder.TakeResult();
  DenseAttr dense_attr(buffer.data() + offset);
  if (attrs_.Set(attr_name, dense_attr)) return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrTensor failed");
}

tensorflow::Status OperationInterface::SetAttrStringList(
    const char* attr_name, const void* const* values, const size_t* lengths,
    int num_values) {
  std::vector<tensorflow::StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = tensorflow::StringPiece(static_cast<const char*>(values[i]),
                                   lengths[i]);
  }
  fallback_attrs_.Set(attr_name, v);

  tfrt::BefAttrEncoder encoder;
  const size_t offset =
      encoder.EncodeStringListAttr(values, lengths, num_values);
  auto buf = encoder.TakeResult();
  tfrt::AggregateAttr aggr_attr(buf.data() + offset);
  // TODO(tfrt-devs): Avoid the copy.
  if (attrs_.Set(attr_name, aggr_attr)) return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrStringList failed");
}

tensorflow::Status OperationInterface::SetAttrFloatList(const char* attr_name,
                                                        const float* values,
                                                        int num_values) {
  fallback_attrs_.Set(
      attr_name, tensorflow::gtl::ArraySlice<const float>(values, num_values));

  if (attrs_.SetArray(attr_name, tfrt::ArrayRef<float>(values, num_values)))
    return ::tensorflow::OkStatus();
  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFloatList failed");
}

tensorflow::Status OperationInterface::SetAttrIntList(const char* attr_name,
                                                      const int64_t* values,
                                                      int num_values) {
  fallback_attrs_.Set(
      attr_name, tensorflow::gtl::ArraySlice<const int64_t>(
                     reinterpret_cast<const int64_t*>(values), num_values));

  if (attrs_.SetArray(attr_name, tfrt::ArrayRef<int64_t>(values, num_values)))
    return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrIntList failed");
}

tensorflow::Status OperationInterface::SetAttrTypeList(
    const char* attr_name, const tensorflow::DataType* values, int num_values) {
  fallback_attrs_.Set(attr_name,
                      tensorflow::gtl::ArraySlice<const tensorflow::DataType>(
                          values, num_values));
  // Convert to OpAttrType first.
  llvm::SmallVector<tfrt::DType, 4> tfrt_dtypes;
  tfrt_dtypes.reserve(num_values);
  for (int i = 0; i < num_values; ++i) {
    tfrt_dtypes.push_back(
        tensorflow::tfd::ConvertTfDataTypeToBefAttrType(values[i]));
  }

  if (attrs_.SetRaw(attr_name, tfrt_dtypes.data(), tfrt::OpAttrType::DTYPE,
                    num_values, OpAttrsRawEntryType::kArray))
    return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrTypeList failed");
}

tensorflow::Status OperationInterface::SetAttrBoolList(
    const char* attr_name, const unsigned char* values, int num_values) {
  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  fallback_attrs_.Set(
      attr_name, tensorflow::gtl::ArraySlice<const bool>(b.get(), num_values));

  // Convert to bool first.
  llvm::SmallVector<bool, 4> bool_array;
  bool_array.reserve(num_values);
  for (int i = 0; i < num_values; ++i) {
    bool_array.push_back(static_cast<bool>((values[i])));
  }
  if (attrs_.SetArray(attr_name,
                      tfrt::ArrayRef<bool>(bool_array.data(), num_values)))
    return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrBoolList failed");
}

tensorflow::Status OperationInterface::SetAttrShapeList(const char* attr_name,
                                                        const int64_t** dims,
                                                        const int* num_dims,
                                                        int num_values) {
  std::unique_ptr<tensorflow::TensorShapeProto[]> proto(
      new tensorflow::TensorShapeProto[num_values]);
  for (int i = 0; i < num_values; ++i) {
    const auto num_dims_i = num_dims[i];

    if (num_dims_i > tensorflow::TensorShape::MaxDimensions()) {
      return tensorflow::errors::InvalidArgument(
          StrCat("Value specified for `", attr_name, "` has ", num_dims_i,
                 " dimensions which is over the limit of ",
                 tensorflow::TensorShape::MaxDimensions(), "."));
    }
    if (num_dims_i < 0) {
      proto[i].set_unknown_rank(true);
    } else {
      const int64_t* dims_i = dims[i];
      auto proto_i = &proto[i];
      for (int d = 0; d < num_dims_i; ++d) {
        proto_i->add_dim()->set_size(dims_i[d]);
      }
    }
  }
  fallback_attrs_.Set(attr_name,
                      tensorflow::gtl::ArraySlice<tensorflow::TensorShapeProto>(
                          proto.get(), num_values));

  BefAttrEncoder encoder;
  const size_t offset = encoder.EncodeShapeListAttr(dims, num_dims, num_values);
  auto buf = encoder.TakeResult();
  tfrt::AggregateAttr aggr_attr(buf.data() + offset);
  if (attrs_.Set(attr_name, aggr_attr)) return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrShapeList failed");
}

tensorflow::Status OperationInterface::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
  size_t num_values = values.size();
  std::vector<const void*> func_attrs(num_values);
  std::vector<size_t> lengths(num_values);

  for (int i = 0; i < num_values; ++i) {
    auto* value_operation = down_cast<const OperationInterface*>(values[i]);
    lengths[i] = value_operation->Name().length();
    func_attrs[i] = value_operation->Name().c_str();
  }

  // Encode the array of function attributes with BEF typed attribute encoder to
  // an aggregated attribute.
  BefAttrEncoder encoder;
  const size_t offset =
      encoder.EncodeFuncListAttr(func_attrs.data(), lengths.data(), num_values);
  auto buf = encoder.TakeResult();
  tfrt::AggregateAttr aggr_attr(buf.data() + offset);
  if (attrs_.Set(attr_name, aggr_attr)) return ::tensorflow::OkStatus();

  return tensorflow::errors::Internal(
      "OperationInterface::SetAttrFunctionList failed");
}

tensorflow::Status OperationInterface::InputLength(const char* input_name,
                                                   int* length) {
  return tensorflow::errors::Unimplemented(
      "Unimplemented OperationInterface::InputLength");
}

tensorflow::Status OperationInterface::OutputLength(const char* output_name,
                                                    int* length) {
  return tensorflow::errors::Unimplemented(
      "Unimplemented OperationInterface::OutputLength");
}

const tensorflow::AbstractOpAttrs* OperationInterface::GetOpAttrs() const {
  return &op_attrs_;
}

void OperationInterface::AddAttrs(const tensorflow::AbstractOpAttrs* op_attrs) {
  auto* tfrt_op_attrs = down_cast<const OpAttrsInterface*>(op_attrs);
  tfrt_op_attrs->GetAttrs()->IterateEntries(
      [this](const OpAttrsRawEntry& entry) {
        attrs_.SetRaw(entry.name, entry.GetData(), entry.type,
                      entry.element_count, entry.entry_type);
      });
  fallback_attrs_.CopyAttributes(*tfrt_op_attrs->GetFallbackAttrs());
}

void OperationInterface::MaybeInferInputAttrs() {
  if (!op_def_) return;
  for (int i = 0; i < args_.size(); i++) {
    auto* handle = args_[i].get();
    const auto& input_def = op_def_->input_arg(i);
    if (!input_def.number_attr().empty() ||
        !input_def.type_list_attr().empty()) {
      // Some clients that are still setting their input attributes manually are
      // adding input list to their op by calling `TFE_OpAddInput` for each of
      // its elements instead of calling `TFE_OpAddInputList`. When this
      // happens, we cannot detect the end of such list, thus lose track of the
      // input arguments in the op definition. To guarantee backward
      // compatibility with those clients, disable automatic inference in this
      // case.
      return;
    }
    const std::string& type_attr = input_def.type_attr();
    if (!type_attr.empty()) {
      bool success = attrs_.Set(
          type_attr, tfrt::GetOpAttrTypeFromDType(
                         tensorflow::tfd::ConvertTfDataTypeToBefAttrType(
                             handle->DataType())));
      if (success) {
        fallback_attrs_.Set(type_attr, handle->DataType());
      }
    }
  }
}

}  // namespace tf
}  // namespace tfrt
