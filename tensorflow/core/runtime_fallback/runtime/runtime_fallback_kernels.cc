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

// This file implements kernels for running TFRT ops/kernels via TF eager
// execution.

#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_kernels.h"

#include <utility>

#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tfrt/cpu/core_runtime/cpu_op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/execute_op_impl.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/attribute_utils.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_frame.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/host_context/sync_kernel_frame.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_serialize_utils.h"  // from @tf_runtime
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_gpu_allocator.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {
namespace tfd {
namespace {
constexpr char kHostContextPtrAttrName[] = "host_ptr";
constexpr char kDefaultCpuDevice[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

}  // namespace

using tfrt::AggregateAttr;
using tfrt::Argument;
using tfrt::AsyncValue;
using tfrt::AsyncValueRef;
using tfrt::BEFAttributeType;
using tfrt::Chain;
using tfrt::DenseAttr;
using tfrt::DenseHostTensor;
using tfrt::ExecutionContext;
using tfrt::Expected;
using tfrt::FuncAttr;
using tfrt::HostBuffer;
using tfrt::HostContext;
using tfrt::KernelErrorHandler;
using tfrt::OpAttrs;
using tfrt::OpAttrsRawEntry;
using tfrt::OpAttrsRef;
using tfrt::OpAttrType;
using tfrt::raw_ostream;
using tfrt::RCReference;
using tfrt::RemainingArguments;
using tfrt::RemainingAttributes;
using tfrt::RemainingResults;
using tfrt::Result;
using tfrt::ShapeAttr;
using tfrt::string_view;
using tfrt::StringAttr;
using tfrt::StringAttribute;
using tfrt::Tensor;
using tfrt::TensorShape;

#define TFD_REPORT_AND_RETURN_IF_ERROR(handler, status) \
  if (!status.ok()) {                                   \
    handler.ReportError(status.error_message());        \
    return;                                             \
  }

// Create RuntimeFallbackTensor from tensorflow::TensorHandle.
// Takes ownership of TensorHandle.
static AsyncValueRef<RuntimeFallbackTensor> CreateRuntimeFallbackTensor(
    TensorHandle* handle, HostContext* host) {
  OwnedTensorHandle th(handle);
  int rank;
  tensorflow::Status status = th->NumDims(&rank);
  if (!status.ok())
    return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(
        "error getting rank from TF tensor handle: ", status.error_message()));

  llvm::SmallVector<tfrt::Index, 4> dims;
  for (auto i = 0; i < rank; ++i) {
    int64_t dim;
    status = th->Dim(i, &dim);
    if (!status.ok())
      return tfrt::MakeErrorAsyncValueRef(
          tfrt::StrCat("error getting dimension from TFE tensor handle: ",
                       status.error_message()));
    dims.push_back(dim);
  }

  TensorShape shape{dims};
  DataType dtype = th->DataType();
  return tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
      shape, GetTfrtDtype(dtype), std::move(th));
}

// Kernel for moving DHT to RuntimeFallbackTensor. Note that the buffer of the
// argument dht is moved to return RuntimeFallbackTensor.
//
// Example usage in MLIR:
//
// %tft, %c2 = "tfd.move_dht_to_tft"(%dht, %c1) :
//   (!dht.dense_host_tensor.i32.2, !hex.chain) -> (!tfd.tf_tensor, !hex.chain)
static std::pair<RuntimeFallbackTensor, Chain> TfdMoveDHTToTFT(
    Argument<DenseHostTensor> dht, Argument<Chain> in_chain,
    const ExecutionContext& exec_ctx) {
  return std::make_pair(
      MoveDHTToRuntimeFallbackTensor(std::move(dht.get()), exec_ctx.host()),
      in_chain.get());
}

// Kernel for converting DHT to RuntimeFallbackTensor.
//
// Example usage in MLIR:
//
// %dht, %c2 = "tfd.convert_tft_to_dht"(%tft, %c1) :
//   (!tfd.tf_tensor,!hex.chain) -> (!dht.dense_host_tensor.i32.2, !hex.chain)
static void TfdConvertTFTToDHT(Argument<RuntimeFallbackTensor> tft,
                               Argument<Chain> in_chain,
                               Result<DenseHostTensor> dht,
                               Result<Chain> out_chain,
                               KernelErrorHandler handler,
                               const ExecutionContext& exec_ctx) {
  dht.Set(tfrt::ConvertTensorOnHost(exec_ctx, tft.get(),
                                    DenseHostTensor::kTensorType)
              .ReleaseRCRef());
  out_chain.Set(in_chain);
}

// Kernel for printing RuntimeFallbackTensor.
//
// Example usage in MLIR:
//
// %c2 = "tfd.print_tft"(%tft, %c1) : (!tfd.tf_tensor, !hex.chain) -> !hex.chain
// TODO(fishx): Remove this kernel and reuse dht.print_tensor.
static void TfdPrintTFT(Argument<RuntimeFallbackTensor> tft,
                        Argument<Chain> in_chain, Result<Chain> out_chain) {
  llvm::outs() << tft.get() << "\n";
  llvm::outs().flush();
  out_chain.Set(in_chain);
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

static tensorflow::Status InjectTfGpuResourcesHelper(
    tensorflow::EagerContext* ctx) {
  // Inject TF's GPU resources to TFRT GpuOpHandler.
  // Note that this requires RuntimeFallbackOpHandler to be created and
  // initialized before tfrt::GpuOpHandler to work.

  auto tf_gpu_process_state = tensorflow::GPUProcessState::singleton();
  if (tf_gpu_process_state && tf_gpu_process_state->HasGPUDevice()) {
    constexpr char gpu_device_type[] = "GPU";
    int num_gpu = ctx->local_device_mgr()->NumDeviceType(gpu_device_type);
    for (int gpu_ordinal = 0; gpu_ordinal < num_gpu; gpu_ordinal++) {
      auto gpu_device_name = absl::StrCat(gpu_device_type, ":", gpu_ordinal);
      Device* device;
      TF_RETURN_IF_ERROR(
          ctx->local_device_mgr()->LookupDevice(gpu_device_name, &device));
      auto gpu_device = static_cast<tensorflow::BaseGPUDevice*>(device);
      if (!gpu_device)
        return tensorflow::errors::NotFound("TF BaseGPUDevice not found");
#if TENSORFLOW_USE_ROCM
      static_assert(
          false,
          "static_cast to GpuContext and CUstream are invalid for ROCm.");
#endif
      CUcontext gpu_context =
          static_cast<stream_executor::gpu::GpuContext*>(
              gpu_device->executor()->implementation()->GpuContextHack())
              ->context();

      // TF GPU allocator is already created in
      // tensorflow::DeviceFactory::AddDevices above, so this GetGPUAllocator
      // ignores options and total_bytes passed in and retrieves allocator based
      // on `tf_device_id`.
      TfDeviceId tf_device_id{gpu_ordinal};
      GPUOptions dummy_options;
      tensorflow::Allocator* tf_allocator =
          tf_gpu_process_state->GetGPUAllocator(dummy_options, tf_device_id,
                                                /*total_bytes=*/0,
                                                /*peer_gpu_ids=*/{});
      if (!tf_allocator)
        return tensorflow::errors::NotFound("TF allocator not found");
      auto accelerator_device_info =
          gpu_device->tensorflow_accelerator_device_info();
      if (!accelerator_device_info)
        return tensorflow::errors::NotFound(
            "accelerator_device_info not found");

      tfrt::gpu::GpuResources gpu_resources;
      gpu_resources.gpu_context = tfrt::gpu::wrapper::Context(gpu_context);
      gpu_resources.allocator_factory =
          CreateRuntimeFallbackGpuAllocatorFactory(tf_allocator);
      gpu_resources.stream = tfrt::gpu::wrapper::Stream(static_cast<CUstream>(
          accelerator_device_info->stream->implementation()->GpuStreamHack()));
      auto platform = tfrt::gpu::wrapper::Platform::CUDA;
      tfrt::gpu::SetTfrtGpuResources(
          tfrt::gpu::wrapper::Device(gpu_ordinal, platform), gpu_resources);
    }
  }
  return OkStatus();
}

tensorflow::Status InjectTfGpuResources() {
  // TODO(zhangqiaorjc) Use more direct and low-level APIs to initialize GPU
  // resources than using EagerContext. Note that this EagerContext is strictly
  // locally scoped and an implementation detail of injecting GPU resources, and
  // not is the same EagerContext set in RequestContext.
  static bool already_injected_gpu_devices = false;
  static absl::Mutex* mutex = new absl::Mutex();

  absl::MutexLock lock(mutex);
  if (!already_injected_gpu_devices) {
    tfrt::Expected<OwnedEagerContext> ctx = InitEagerContext();
    if (!ctx) {
      return tensorflow::errors::Internal(
          tfrt::StrCat("error initializing eager context: ", ctx.takeError()));
    }

    // GPU resources should be injected once per gpu ordinal.
    TF_RETURN_IF_ERROR(InjectTfGpuResourcesHelper(ctx->get()));
    already_injected_gpu_devices = true;
  }

  return OkStatus();
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Kernel for initializing TF EagerContext.
//
// This kernel should be invoked at least once before any TF delegation kernels
// are invoked. Redundant calls to initialize the eager context are skipped.
//
// Example usage in MLIR:
//
// %c2 = "tfd.init_eager_context"(%c1): (!hex.chain) -> !hex.chain
//
static void TfdInitEagerContext(Argument<Chain> in_chain,
                                Result<Chain> out_chain,
                                KernelErrorHandler handler,
                                const ExecutionContext& exec_ctx) {
  tfrt::ResourceContext* resource_context = exec_ctx.resource_context();
  tensorflow::tfd::EagerContextResource* eager_context_resource =
      resource_context
          ->GetOrCreateResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName);
  (void)eager_context_resource;

  // TODO(zhangqiaorjc): Inject GPU resources to GPU kernels.
  out_chain.Set(in_chain);
}

OwnedTFTensor MoveDHTToTFTensor(DenseHostTensor&& dht, HostContext* host) {
  llvm::SmallVector<tfrt::Index, 4> dims;
  dht.shape().GetDimensions(&dims);

  HostBuffer* host_buffer = dht.ReleaseBuffer().release();
  auto deallocator = [](void* data, size_t len, void* arg) {
    auto* host_buffer = reinterpret_cast<HostBuffer*>(arg);
    host_buffer->DropRef();
  };

  CheckBoolCompatibility();
  OwnedTFTensor tf_tensor{
      TF_NewTensor(static_cast<TF_DataType>(GetTfDataType(dht.dtype())),
                   dims.data(), dims.size(), host_buffer->data(),
                   host_buffer->size(), deallocator, host_buffer)};
  return tf_tensor;
}

static tensorflow::Status DecodeDenseAttrToTensorInterface(
    const DenseAttr& dense_attr, HostContext* host,
    tensorflow::TensorInterface* result) {
  Expected<DenseHostTensor> dht =
      tfrt::DeserializeDenseHostTensorFromDenseAttr(dense_attr, host);
  if (!dht)
    return tensorflow::errors::Internal(tfrt::StrCat(
        "cannot create DenseHostTensor in DecodeDenseAttrToTensorInterface:",
        dht.takeError()));
  OwnedTFTensor tf_tensor = MoveDHTToTFTensor(std::move(*dht), host);
  tensorflow::Tensor t;
  TF_RETURN_IF_ERROR(TF_TensorToTensor(tf_tensor.get(), &t));
  *result = tensorflow::TensorInterface(std::move(t));
  return OkStatus();
}

// Handle attributes.
//
// Refer to tensorflow/core/framework/attr_value.proto and
// tensorflow/c/eager/c_api.h.
//
// Note we currently do not support the following attribute value types:
// TFE_OpSetAttrFunction
// TFE_OpSetAttrFunctionName
static tensorflow::Status PrepareAttributes(EagerOperation* eager_op,
                                            const OpAttrsRef& attrs,
                                            HostContext* host,
                                            EagerContext* eager_ctx) {
  tensorflow::Status status;
  attrs.IterateEntries([eager_op, eager_ctx, status_ptr = &status, host,
                        &attrs](const OpAttrsRawEntry& entry) {
    // TFE does not expect a device attribute.
    assert(strcmp(entry.name, "device") != 0);
    if (IsUnusedAttribute(entry.name)) {
      return;
    } else if (entry.IsArray()) {
      if (entry.element_count == 0) {
        if (entry.type == OpAttrType::CHAR) {
          // Empty string.
          std::string empty_str;
          *status_ptr = eager_op->SetAttrString(entry.name, empty_str.data(),
                                                empty_str.size());
        } else {
          // Empty array of other types.
          AttrValue empty_attr_value;
          eager_op->MutableAttrs()->Set(entry.name, empty_attr_value);
        }
      } else if (entry.type == OpAttrType::CHAR) {
        string_view attr_value = attrs.GetStringAsserting(entry.name);
        *status_ptr = eager_op->SetAttrString(entry.name, attr_value.data(),
                                              attr_value.size());
      } else if (entry.type == OpAttrType::FUNC) {
        string_view attr_value = attrs.GetFuncNameAsserting(entry.name);
        *status_ptr = eager_op->SetAttrFunctionName(
            entry.name, attr_value.data(), attr_value.size());
      } else if (entry.type == OpAttrType::I64) {
        llvm::ArrayRef<int64_t> int_array =
            attrs.GetArrayAsserting<int64_t>(entry.name);
        *status_ptr = eager_op->SetAttrIntList(entry.name, int_array.data(),
                                               int_array.size());
      } else if (entry.type == OpAttrType::F32) {
        llvm::ArrayRef<float> float_array =
            attrs.GetArrayAsserting<float>(entry.name);
        *status_ptr = eager_op->SetAttrFloatList(entry.name, float_array.data(),
                                                 float_array.size());
      } else if (entry.type == OpAttrType::BOOL) {
        llvm::ArrayRef<bool> bool_array =
            attrs.GetArrayAsserting<bool>(entry.name);
        // SetAttrBoolList expects const unsigned char*, not const bool*.
        std::vector<unsigned char> bool_char_array(bool_array.begin(),
                                                   bool_array.end());
        *status_ptr = eager_op->SetAttrBoolList(
            entry.name, bool_char_array.data(), bool_char_array.size());
      } else if (entry.type == OpAttrType::DTYPE) {
        const auto& op_attr = attrs.GetRawAsserting(entry.name);
        assert(op_attr.IsArray());

        // DTypes in BEF attributes are tfrt::DType enums. So we need
        // to convert then to tensorflow data types first.
        auto bef_dtypes = llvm::makeArrayRef(
            static_cast<const tfrt::DType*>(op_attr.GetData()),
            op_attr.element_count);

        llvm::SmallVector<tensorflow::DataType, 4> tf_dtypes;
        tf_dtypes.reserve(bef_dtypes.size());
        for (auto bef_dtype : bef_dtypes) {
          tf_dtypes.push_back(ConvertBefAttrTypeToTfDataType(bef_dtype));
        }

        *status_ptr = eager_op->SetAttrTypeList(entry.name, tf_dtypes.data(),
                                                tf_dtypes.size());
      } else {
        *status_ptr =
            tensorflow::errors::Internal("unsupported array attribute type");
      }
    } else {
      if (entry.type == OpAttrType::I64) {
        int64_t attr_value = attrs.GetAsserting<int64_t>(entry.name);
        *status_ptr = eager_op->SetAttrInt(entry.name, attr_value);
      } else if (entry.type == OpAttrType::F32) {
        float attr_value = attrs.GetAsserting<float>(entry.name);
        *status_ptr = eager_op->SetAttrFloat(entry.name, attr_value);
      } else if (entry.type == OpAttrType::BOOL) {
        bool attr_value = attrs.GetAsserting<bool>(entry.name);
        *status_ptr = eager_op->SetAttrBool(entry.name, attr_value);
      } else if (entry.type == OpAttrType::DTYPE) {
        OpAttrType op_attr_type = attrs.GetAsserting<OpAttrType>(entry.name);
        DataType tf_dtype = ConvertToTfDataType(op_attr_type);
        *status_ptr = eager_op->SetAttrType(entry.name, tf_dtype);
      } else if (entry.type == OpAttrType::SHAPE) {
        tfrt::ShapeAttr shape_attr =
            attrs.GetAsserting<tfrt::ShapeAttr>(entry.name);
        if (shape_attr.HasRank()) {
          *status_ptr = eager_op->SetAttrShape(
              entry.name, shape_attr.GetShape().data(), shape_attr.GetRank());
        } else {
          *status_ptr = eager_op->SetAttrShape(entry.name, /*dims=*/nullptr,
                                               /*num_dims=*/-1);
        }
      } else if (entry.type == OpAttrType::DENSE) {
        DenseAttr dense_attr = attrs.GetAsserting<DenseAttr>(entry.name);
        tensorflow::TensorInterface interface;
        *status_ptr =
            DecodeDenseAttrToTensorInterface(dense_attr, host, &interface);
        if (!status_ptr->ok()) return;
        *status_ptr = eager_op->SetAttrTensor(entry.name, &interface);
      } else if (entry.type == OpAttrType::AGGREGATE) {
        AggregateAttr list_attr = attrs.GetAsserting<AggregateAttr>(entry.name);
        int num_values = list_attr.GetNumElements();

        // Insert a dummy list attribute to the NodeDef if the aggregate attr
        // is empty. This is needed because the ValidateNodeDef method checks
        // the encoded_attr_ map for expected attributes, specified in the
        // OpDef.
        if (num_values == 0) {
          // The int type is just a placeholder and doesn't matter.
          std::vector<int> dummy_attr;
          eager_op->MutableAttrs()->Set(
              entry.name, gtl::ArraySlice<const int>(dummy_attr.data(), 0));
          return;
        }

        // It is guaranteed that items in one list attribute have the same
        // type, though their sizes can be different. In particular,
        // list(TensorShape) and list(Tensor) attribute types have to be
        // encoded as AggregateAttr.
        auto attr_base = list_attr.GetAttribute(0);
        if (IsDataTypeAttribute(attr_base.type()) &&
            GetDataType(attr_base.type()) == tfrt::DType::String) {
          // Handle list(string).
          llvm::SmallVector<const void*, 8> values;
          llvm::SmallVector<size_t, 8> lengths;
          values.reserve(num_values);
          lengths.reserve(num_values);
          for (int i = 0; i < num_values; ++i) {
            auto string_attr = list_attr.GetAttributeOfType<StringAttr>(i);
            values.push_back(string_attr.GetValue().data());
            lengths.push_back(string_attr.GetValue().size());
          }
          *status_ptr = eager_op->SetAttrStringList(entry.name, values.data(),
                                                    lengths.data(), num_values);
        } else if (IsFuncAttribute(attr_base.type())) {
          std::vector<const AbstractOperation*> funcs(num_values);
          for (int i = 0; i < num_values; ++i) {
            auto func_attr = list_attr.GetAttributeOfType<FuncAttr>(i);
            // TODO(chuanhao): Creating a EagerOperation here is expensive.
            // consider using AttrBuilder to set attribute directly.
            ImmediateExecutionOperation* new_op = eager_ctx->CreateOperation();
            auto func_name = func_attr.GetFunctionName();
            *status_ptr = new_op->Reset(func_name.str().c_str(),
                                        /*raw_device_name=*/nullptr);
            funcs[i] = new_op;
          }
          *status_ptr =
              eager_op->SetAttrFunctionList(entry.name, absl::MakeSpan(funcs));
        } else if (attr_base.type() == BEFAttributeType::kShape) {
          // Handle list(TensorShape).
          llvm::SmallVector<int, 8> ranks;
          llvm::SmallVector<const int64_t*, 8> dims;
          ranks.reserve(num_values);
          dims.reserve(num_values);
          for (int i = 0; i < num_values; ++i) {
            auto shape_attr = list_attr.GetAttributeOfType<ShapeAttr>(i);
            if (shape_attr.HasRank()) {
              ranks.push_back(shape_attr.GetRank());
              dims.push_back(shape_attr.GetShape().data());
            } else {
              ranks.push_back(-1);
              dims.push_back(nullptr);
            }
          }
          *status_ptr = eager_op->SetAttrShapeList(entry.name, dims.data(),
                                                   ranks.data(), num_values);
        } else {
          *status_ptr =
              tensorflow::errors::Internal("unsupported list attribute type");
        }
      } else {
        *status_ptr =
            tensorflow::errors::Internal("unsupported scalar attribute type");
      }
    }
  });
  return status;
}

Status CallEagerExecute(const tfrt::ExecutionContext& exec_ctx,
                        EagerContext* eager_ctx, const char* op_name,
                        const char* device_name,
                        llvm::ArrayRef<TensorHandle*> input_tensor_handles,
                        const OpAttrsRef& attrs,
                        llvm::MutableArrayRef<tensorflow::AbstractTensorHandle*>
                            result_tensor_handles) {
  assert(eager_ctx != nullptr && "EagerContext is NULL");

  // Create TF EagerOperation.
  OwnedEagerOperation eager_op{new EagerOperation(eager_ctx)};
  TF_RETURN_IF_ERROR(eager_op->Reset(op_name, device_name));

  // Handle inputs.
  for (TensorHandle* input_tensor : input_tensor_handles) {
    TF_RETURN_IF_ERROR(eager_op->AddInput(input_tensor));
  }

  // Handle attributes.
  auto* host = exec_ctx.host();
  TF_RETURN_IF_ERROR(PrepareAttributes(eager_op.get(), attrs, host, eager_ctx));

  int num_retvals = result_tensor_handles.size();
  TF_RETURN_IF_ERROR(eager_op->Execute(
      absl::MakeSpan(result_tensor_handles.data(), num_retvals), &num_retvals));

  return OkStatus();
}

static bool ShouldAddHostContextAttr(const char* op_name) {
  // NOTE(rachelim): In the future, if more ops require this, instead of
  // checking against a whitelist of op names, we could check whether the op
  // contains an attribute called `host_ptr`.
  return strcmp(op_name, "TFRTMakeIterator") == 0;
}

// TODO(zhangqiaorjc): Unify implementation with RuntimeFallbackKernel.
AsyncValueRef<Chain> RuntimeFallbackExecute(
    const tfrt::ExecutionContext& exec_ctx, EagerContext* eager_ctx,
    const char* op_name, const char* device_name,
    llvm::ArrayRef<Tensor*> arguments, const OpAttrsRef& attrs,
    llvm::MutableArrayRef<RCReference<AsyncValue>> results) {
  auto emit_error = [&exec_ctx, results](const tensorflow::Status& status) {
    // Set the correct TFRT error code according to the error propagated from
    // runtime fallback execution.
    auto error =
        EmitErrorAsync(exec_ctx, status.error_message(),
                       tfrt::ConvertTfErrorCodeToTfrtErrorCode(status));
    // Set all results to error.
    std::fill(results.begin(), results.end(), error);
    return error;
  };

  llvm::SmallVector<TensorHandle*, 4> input_tensor_handles;
  input_tensor_handles.reserve(arguments.size());
  for (Tensor* input_tensor : arguments) {
    input_tensor_handles.push_back(
        llvm::cast<RuntimeFallbackTensor>(input_tensor)->GetTensorHandle());
  }

  int num_retvals = results.size();
  llvm::SmallVector<tensorflow::AbstractTensorHandle*, 4> result_tensor_handles(
      num_retvals);
  Status status;
  if (!ShouldAddHostContextAttr(op_name)) {
    status =
        CallEagerExecute(exec_ctx, eager_ctx, op_name, device_name,
                         input_tensor_handles, attrs, result_tensor_handles);
  } else {
    // Wrap the HostContext pointer in an attribute. This is necessary for
    // TF ops that require the TFRT HostContext to function. These kernels
    // should not create their own HostContexts.
    // TODO(rachelim): Support copying over non-host_ptr attrs, if there are
    // any.
    assert(attrs.GetNumEntries() == 1);
    OpAttrs updated;

    updated.Set(kHostContextPtrAttrName,
                reinterpret_cast<int64_t>(exec_ctx.host()));
    status = CallEagerExecute(
        exec_ctx, eager_ctx, op_name, device_name, input_tensor_handles,
        OpAttrsRef(std::move(updated)), result_tensor_handles);
  }
  if (!status.ok()) return emit_error(status);

  auto host = exec_ctx.host();
  for (int i = 0; i < num_retvals; ++i) {
    auto expected_fallback_tensor =
        CreateRuntimeFallbackTensorFromTfTensorHandle(
            OwnedTensorHandle{
                TensorHandleFromInterface(result_tensor_handles[i])},
            host);
    if (!expected_fallback_tensor)
      results[i] = EmitErrorAsync(
          exec_ctx, tfrt::StrCat(expected_fallback_tensor.takeError()));
    else
      results[i] = tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
          std::move(*expected_fallback_tensor));
  }

  return tfrt::GetReadyChain();
}

AsyncValueRef<Chain> RuntimeFallbackExecute(
    const tfrt::ExecutionContext& exec_ctx, const char* op_name,
    const char* device_name, llvm::ArrayRef<Tensor*> arguments,
    const OpAttrsRef& attrs,
    llvm::MutableArrayRef<RCReference<AsyncValue>> results) {
  // Get EagerContext.
  auto eager_ctx_expected = GetEagerContext(exec_ctx);
  if (!eager_ctx_expected) {
    auto error = EmitErrorAsync(exec_ctx, eager_ctx_expected.takeError(),
                                tfrt::ErrorCode::kUnknown);
    // Set all results to error.
    std::fill(results.begin(), results.end(), error);
    return std::move(error);
  }
  EagerContext* eager_ctx = eager_ctx_expected.get();

  return RuntimeFallbackExecute(exec_ctx, eager_ctx, op_name, device_name,
                                arguments, attrs, results);
}

// Kernel to delegate to the current TF runtime kernel.
//
// Example usage in MLIR:
//
// %c2, %tft_c = "tfd.delegate_kernel"(%c1, %tft_a, %tft_b) {op_name = "MatMul"}
// : (!hex.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (!hex.chain,
// !tfd.tf_tensor)
// TODO(jingdong): Enqueue the TFE kernel execution as blocking task to the
// ConcurrentWorkQueue.
static void RuntimeFallbackKernel(
    Argument<Chain> in_chain, RemainingArguments input_tensors,
    Result<Chain> out_chain, RemainingResults output_tensors,
    StringAttribute op_name, RemainingAttributes remaining_attributes,
    KernelErrorHandler handler, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  tfrt::ResourceContext* resource_context = exec_ctx.resource_context();
  EagerContextResource* eager_context_resource =
      resource_context->GetOrCreateResource<EagerContextResource>(
          tensorflow::tfd::kEagerContextResourceName);
  tfrt::Expected<EagerContext*> eager_ctx_expected =
      eager_context_resource->GetTFEagerContext();
  if (!eager_ctx_expected) {
    handler.ReportError("eager_ctx_expected.takeError()");
    return;
  }
  EagerContext* eager_ctx = eager_ctx_expected.get();

  // Construct TF EagerOperation.
  // Need to copy op_name to a std::string to ensure the string is
  // null-terminated.
  std::string op_name_str = [&] {
    auto view = op_name.get();
    view.consume_front("tf.");
    return view.str();
  }();

  OwnedEagerOperation eager_op{new EagerOperation(eager_ctx)};
  TFD_REPORT_AND_RETURN_IF_ERROR(
      handler,
      eager_op->Reset(op_name_str.c_str(), /*raw_device_name=*/nullptr));

  // Handle inputs.
  for (AsyncValue* input_tensor_av : input_tensors.values()) {
    auto input_tensor_handle =
        input_tensor_av->get<RuntimeFallbackTensor>().GetTensorHandle();
    TFD_REPORT_AND_RETURN_IF_ERROR(handler,
                                   eager_op->AddInput(input_tensor_handle));
  }

  // Handle TF op attributes.
  // TODO(zhangqiaorjc): Encode TF attributes using native MLIR attribute types.
  assert(remaining_attributes.size() % 2 == 0);
  int num_tf_attrs = remaining_attributes.size() / 2;
  for (int i = 0; i < num_tf_attrs; ++i) {
    // Each TF attribute is represented as a pair of name and value strings.
    // Make a copy for `attr_name` to ensure null-termination.
    std::string attr_name =
        remaining_attributes.GetStringAttribute(i * 2).str();
    absl::string_view attr_value = ToAbslStringView(
        remaining_attributes.GetStringAttribute(i * 2 + 1).get());
    std::vector<absl::string_view> value_split =
        tfd::AttrValueSplit(attr_value);

    // Handle different TF attribute types.
    if (value_split[0] == "string") {
      TFD_REPORT_AND_RETURN_IF_ERROR(
          handler,
          eager_op->SetAttrString(attr_name.c_str(), value_split[1].data(),
                                  value_split[1].size()));
    } else if (value_split[0] == "bool") {
      bool bool_val;
      TFD_REPORT_AND_RETURN_IF_ERROR(
          handler, ParseBoolAttrValue(value_split[1], &bool_val));
      TFD_REPORT_AND_RETURN_IF_ERROR(
          handler, eager_op->SetAttrBool(attr_name.c_str(), bool_val));
    } else if (value_split[0] == "int") {
      int64_t int_val;
      TFD_REPORT_AND_RETURN_IF_ERROR(
          handler, ParseIntAttrValue(value_split[1], &int_val));
      TFD_REPORT_AND_RETURN_IF_ERROR(
          handler, eager_op->SetAttrInt(attr_name.c_str(), int_val));
    } else if (value_split[0] == "tftensor") {
      tensorflow::Tensor t;
      TFD_REPORT_AND_RETURN_IF_ERROR(handler,
                                     ParseTensorAttrValue(value_split[1], &t));
      tensorflow::TensorInterface interface(t);
      TFD_REPORT_AND_RETURN_IF_ERROR(
          handler, eager_op->SetAttrTensor(attr_name.c_str(), &interface));
    } else if (value_split[0] == "tfdtype") {
      DataType dtype;
      TFD_REPORT_AND_RETURN_IF_ERROR(handler,
                                     ParseTfDataType(value_split[1], &dtype));
      TFD_REPORT_AND_RETURN_IF_ERROR(
          handler, eager_op->SetAttrType(attr_name.c_str(), dtype));
    } else if (value_split[0] == "tfshape") {
      std::vector<int64_t> dims;
      TFD_REPORT_AND_RETURN_IF_ERROR(
          handler, ParseTensorShapeAttrValue(value_split[1], &dims));
      TFD_REPORT_AND_RETURN_IF_ERROR(
          handler,
          eager_op->SetAttrShape(attr_name.c_str(), dims.data(), dims.size()));
    } else {
      handler.ReportError("attribute type not yet supported");
      return;
    }
  }

  // Invoke the TF EagerOperation.
  int num_retvals = output_tensors.size();
  llvm::SmallVector<tensorflow::AbstractTensorHandle*, 4> retvals(num_retvals);

  tensorflow::Status status = eager_op->Execute(
      absl::MakeSpan(retvals.data(), num_retvals), &num_retvals);
  TFD_REPORT_AND_RETURN_IF_ERROR(handler, status);

  // Handle outputs.
  if (num_retvals != output_tensors.size()) {
    handler.ReportError("Incorrect number of output values");
    return;
  }
  for (int i = 0; i < num_retvals; ++i) {
    OwnedTensorHandle owned_th{TensorHandleFromInterface(retvals[i])};
    if (!owned_th) handler.ReportError("TensorHandleFromInterface failed");
    auto fallback_tensor = CreateRuntimeFallbackTensorFromTfTensorHandle(
        std::move(owned_th), host);
    if (!fallback_tensor) {
      output_tensors[i] = tfrt::MakeErrorAsyncValueRef(
          tfrt::StrCat(fallback_tensor.takeError()));
    } else {
      output_tensors[i] =
          tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
              std::move(*fallback_tensor));
    }
  }
  out_chain.Set(in_chain);
}

static void EmitErrorAndSetInResults(
    const tfrt::ExecutionContext& exec_ctx,
    const tfrt::DecodedDiagnostic& error,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results) {
  auto error_av = tfrt::EmitErrorAsync(exec_ctx, error.message, error.code);
  std::fill(results.begin(), results.end(), error_av);
}

// Convert the tfrt::TensorHandle to tensorflow::Tensor. `device` is the target
// device for the converted tensorflow::Tensor.
//
// TODO(tfrt-devs): Currently the target device can only be CPU. We need to add
// support for more devices.
void CoreRTTensorHandleToFallbackTensorInternal(
    llvm::ArrayRef<tfrt::AsyncValue*> tensorhandle_args,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>>
        tf_tensor_results,
    tfrt::string_view device, const tfrt::ExecutionContext& exec_ctx) {
  assert(tensorhandle_args.size() == tf_tensor_results.size());

  auto set_result = [&](tfrt::RCReference<tfrt::AsyncValue>& result,
                        llvm::Expected<tensorflow::Tensor> tf_tensor) {
    auto result_ref = tfrt::MakeUnconstructedAsyncValueRef<
        tensorflow::tfrt_stub::FallbackTensor>();
    if (!tf_tensor) {
      result_ref.SetError(tfrt::StrCat(tf_tensor.takeError()));
    } else {
      result_ref.emplace(std::move(tf_tensor.get()));
    }
    result = std::move(result_ref);
  };

  auto maybe_convert_runtime_fallback_tensor =
      [&exec_ctx](
          tfrt::AsyncValueRef<Tensor> tensor_avref,
          const tfrt::Device& src_device,
          const tfrt::Device& dst_device) -> tfrt::AsyncValueRef<tfrt::Tensor> {
    // TODO(tfrt-devs): Remove implicit conversion in this kernel since it will
    // have extra overheads.
    // Convert RuntimeFallbackTensor to KernelFallbackTensor before calling
    // into kernel fallback. That is because today kernel fallback cannot read
    // tensor from runtime fallback tensor because we don't want kernel
    // fallback to depend on runtime fallback.
    assert(tensor_avref.IsAvailable());
    assert(!tensor_avref.IsError());
    auto& tensor = tensor_avref.get();
    if (!tensor.IsTensorType(DenseHostTensor::kTensorType) ||
        !src_device.IsDeviceType(tfrt::CpuDevice::kDeviceType) ||
        !dst_device.IsDeviceType(tfrt::CpuDevice::kDeviceType)) {
      return tfrt::ConvertTensor(exec_ctx, tensor,
                                 /*src=*/src_device,
                                 /*dst=*/dst_device,
                                 KernelFallbackTensor::kTensorType);
    }
    return tensor_avref;
  };

  auto dst_device =
      exec_ctx.host()->GetDeviceManager()->GetDeviceRef<tfrt::Device>(device);

  // Retrieve the underlying pointer of tfrt::Tensor. We don't need to do
  // extra ownership management here because KernelFallbackExecuteCompat()
  // will always convert it to tensorflow::Tensor which is itself refcounted.
  for (int i = 0; i < tensorhandle_args.size(); ++i) {
    if (!dst_device) {
      tf_tensor_results[i] = tfrt::MakeErrorAsyncValueRef(
          tfrt::StrCat("Failed to find device with name ", device));
      continue;
    }
    auto& tensor_handle = tensorhandle_args[i]->get<tfrt::TensorHandle>();
    assert(tensor_handle.IsDeviceAvailable());
    assert(!tensor_handle.IsDeviceError());

    auto* tensor_av = tensor_handle.GetAsyncTensor();
    auto tensor_avref = tfrt::AsyncValueRef<Tensor>(FormRef(tensor_av));

    auto& src_device = *tensor_handle.GetAvailableDevice();
    AsyncValueRef<Tensor> knfb_tensor;
    if (!tensor_av->IsAvailable()) {
      auto ind_av = tfrt::MakeIndirectAsyncValue();
      knfb_tensor = AsyncValueRef<Tensor>(ind_av.CopyRef());
      tensor_av->AndThen(
          [tensor_avref = std::move(tensor_avref), ind_av = std::move(ind_av),
           &src_device, dst_device = dst_device.CopyRef(),
           maybe_convert_runtime_fallback_tensor, exec_ctx]() mutable {
            ind_av->ForwardTo(maybe_convert_runtime_fallback_tensor(
                std::move(tensor_avref), src_device, *dst_device));
          });
    } else {
      knfb_tensor = maybe_convert_runtime_fallback_tensor(
          std::move(tensor_avref), src_device, *dst_device);
    }

    if (!knfb_tensor.IsAvailable()) {
      auto result_ref = tfrt::MakeIndirectAsyncValue();
      tf_tensor_results[i] = result_ref;
      auto knfb_tensor_av = knfb_tensor.GetAsyncValue();
      knfb_tensor_av->AndThen([knfb_tensor = std::move(knfb_tensor),
                               result_ref = std::move(result_ref),
                               dst_device = dst_device.CopyRef(),
                               exec_ctx]() mutable {
        if (knfb_tensor.IsError()) {
          result_ref->ForwardTo(std::move(knfb_tensor));
          return;
        }
        auto expected_tf_tensor =
            TFRTTensorToTFTensor(knfb_tensor.get(), exec_ctx.host());
        if (!expected_tf_tensor) {
          auto error =
              tfrt::EmitErrorAsync(exec_ctx, expected_tf_tensor.takeError());
          result_ref->ForwardTo(std::move(error));
        } else {
          auto tf_tensor_ref = tfrt::MakeAvailableAsyncValueRef<
              tensorflow::tfrt_stub::FallbackTensor>(
              std::move(expected_tf_tensor.get()));
          result_ref->ForwardTo(std::move(tf_tensor_ref));
        }
      });
    } else {
      set_result(tf_tensor_results[i],
                 TFRTTensorToTFTensor(knfb_tensor.get(), exec_ctx.host()));
    }
  }
}

// Returns true if the tensorflow::DataType is trivially copyable.
static bool IsTriviallyCopyableTensorflowDataType(tensorflow::DataType dtype) {
  static const auto* const non_trivially_copyable_dtypes =
      new absl::flat_hash_set<tensorflow::DataType>{
          tensorflow::DataType::DT_STRING, tensorflow::DataType::DT_RESOURCE,
          tensorflow::DataType::DT_VARIANT};
  return !non_trivially_copyable_dtypes->contains(dtype);
}

static llvm::Expected<tensorflow::tfrt_stub::FallbackTensor> ConstDenseTensor(
    tfrt::DenseAttr value, const tfrt::ExecutionContext& context) {
  auto dtype = GetTfDataType(tfrt::DType(value.dtype()));
  // The data type must be trivially copyable so that we can use memcpy.
  DCHECK(IsTriviallyCopyableTensorflowDataType(dtype));
  tensorflow::Tensor tensor(dtype, tensorflow::TensorShape(value.shape()));
  std::memcpy(tensor.data(), value.GetElements(), tensor.TotalBytes());
  return tensorflow::tfrt_stub::FallbackTensor(tensor);
}

static llvm::Expected<tensorflow::tfrt_stub::FallbackTensor> ConstStringTensor(
    tfrt::ArrayAttr shape, tfrt::AggregateAttr value,
    const ExecutionContext& context) {
  llvm::SmallVector<int64_t> dims;
  auto tfrt_tensor_shape = tfrt::TensorShape(shape.GetValue<int64_t>());
  tfrt_tensor_shape.GetDimensions(&dims);
  tensorflow::Tensor tensor(tensorflow::DT_STRING,
                            tensorflow::TensorShape(dims));
  auto len = tensor.NumElements();
  auto from = value;
  auto to = tensor.flat<tensorflow::tstring>();
  if (from.GetNumElements() == 1) {
    // All elements are the same, and only one element is saved in BEF.
    for (size_t i = 0; i < len; ++i) {
      to(i) =
          ToAbslStringView(from.GetAttributeOfType<StringAttr>(0).GetValue());
    }
  } else {
    assert(len == from.GetNumElements());
    for (size_t i = 0; i < len; ++i) {
      to(i) =
          ToAbslStringView(from.GetAttributeOfType<StringAttr>(i).GetValue());
    }
  }
  return tensorflow::tfrt_stub::FallbackTensor(tensor);
}

// The BEF kernel for tfrt::TensorHandle to tensorflow::Tensor conversion.
void CoreRTTensorHandleToFallbackTensor(
    RemainingArguments args, RemainingResults results, StringAttr device,
    const tfrt::ExecutionContext& exec_ctx) {
  tensorflow::profiler::TraceMe trace_me(
      "corert_tensorhandle_to_fallback_tensor");
  trace_me.AppendMetadata([request_id = exec_ctx.request_ctx()->id()]() {
    return tensorflow::profiler::TraceMeEncode({{"id", request_id}});
  });

  CoreRTTensorHandleToFallbackTensorInternal(args.values(), results.values(),
                                             device.GetValue(), exec_ctx);
}

// Convert the tensorflow::Tensor to tfrt::TensorHandle. `device` is the device
// for the input tensorflow::Tensor.
//
// TODO(tfrt-devs): Currently the input device can only be CPU. We need to add
// support for more devices.
static void FallbackTensorToCoreRTTensorHandleInternal(
    llvm::ArrayRef<tfrt::AsyncValue*> tf_tensor_args,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>>
        tensorhandle_results,
    absl::string_view device, const tfrt::ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();

  assert(tf_tensor_args.size() == tensorhandle_results.size());
  for (int i = 0; i < tf_tensor_args.size(); ++i) {
    auto* av = tf_tensor_args[i];
    auto& tf_tensor = av->get<tensorflow::tfrt_stub::FallbackTensor>().tensor();
    AsyncValueRef<tfrt::Tensor> kernel_fallback_tensor =
        tfrt::MakeAvailableAsyncValueRef<KernelFallbackTensor>(tf_tensor);
    auto metadata = kernel_fallback_tensor.get().metadata();

    tensorhandle_results[i] =
        tfrt::MakeAvailableAsyncValueRef<tfrt::TensorHandle>(
            host->GetDeviceManager()->GetDeviceRef<tfrt::Device>(
                {device.data(), device.size()}),
            metadata, std::move(kernel_fallback_tensor));
  }
}

// The BEF kernel for tensorflow::Tensor to tfrt::TensorHandle conversion.
void FallbackTensorToCoreRTTensorHandle(
    RemainingArguments args, RemainingResults results, StringAttr device,
    const tfrt::ExecutionContext& exec_ctx) {
  tensorflow::profiler::TraceMe trace_me(
      "fallback_tensor_to_corert_tensorhandle");
  trace_me.AppendMetadata([request_id = exec_ctx.request_ctx()->id()]() {
    return tensorflow::profiler::TraceMeEncode({{"id", request_id}});
  });

  FallbackTensorToCoreRTTensorHandleInternal(
      args.values(), results.values(), ToAbslStringView(device.GetValue()),
      exec_ctx);
}

static llvm::Expected<bool> Predicate(
    const tensorflow::tfrt_stub::FallbackTensor& input,
    const tfrt::ExecutionContext& exec_ctx) {
  const auto& tensor = input.tensor();
  if (TensorShapeUtils::IsScalar(tensor.shape())) {
    switch (tensor.dtype()) {
#define CASE(T)                  \
  case DataTypeToEnum<T>::value: \
    return tensor.scalar<T>()() != 0;

      CASE(float);
      CASE(double);
      CASE(uint8);
      CASE(int8);
      CASE(int16);
      CASE(int32);
      CASE(int64_t);
      CASE(bool);
#undef CASE
      case DT_STRING:
        return !tensor.scalar<tstring>()().empty();
      default:
        return tfrt::MakeStringError(DataTypeString(tensor.dtype()),
                                     " cannot be converted to a boolean");
    }
  }

  return tensor.NumElements() > 0;
}

tfrt::Chain PrintFallbackTensor(
    const tensorflow::tfrt_stub::FallbackTensor& arg, const tfrt::Chain& ch) {
  std::string message;
  llvm::raw_string_ostream(message) << arg.tensor().DebugString() << "\n";
  printf("%s", message.c_str());
  fflush(stdout);
  return tfrt::Chain();
}

// The legacy kernel implementation that dispatches runtime fallback operations.
// Since the arguments and results are tensorflow::Tensors, internally it
// does conversions between RuntimeFallbackTensor and tensorflow::Tensor.
static void RuntimeFallbackExecuteOp(
    RemainingArguments args, RemainingResults results, StringAttr device_attr,
    AggregateAttr op_attr_array, AggregateAttr op_func_attr_array,
    StringAttr op_name_attr, tfrt::AsyncValueRef<tfrt::Chain>* op_chain,
    const ExecutionContext& exec_ctx) {
  auto set_error = [&exec_ctx, results](tfrt::string_view msg) {
    auto error_av = EmitErrorAsync(exec_ctx, msg, tfrt::ErrorCode::kUnknown);
    // Set all results to error.
    for (int i = 0, n = results.size(); i < n; ++i) results[i] = error_av;
  };

  auto op_name = op_name_attr.GetValue();
  op_name.consume_front("tf.");

  // The device name might not be in the format expected by tensorflow. In that
  // case we change it to the correct format. Currently we only support CPU.
  //
  // TODO(tfrt-devs): Make sure device names passed to fallback kernels are in
  // the tensorflow format.
  std::string device_name = device_attr.GetValue().str();
  if (!absl::StartsWith(device_name, "/")) device_name = kDefaultCpuDevice;

  // Set up OpAttrs.
  tfrt::OpAttrs op_attrs;
  tfrt::SetUpOpAttrs(op_attr_array, &op_attrs);

  // Set up OpAttrs specifically for function attributes.
  tfrt::SetUpOpFuncAttrs(op_func_attr_array, &op_attrs);

  // Get EagerContext.
  auto eager_ctx_expected = GetEagerContext(exec_ctx);
  if (!eager_ctx_expected) {
    set_error(tfrt::StrCat(eager_ctx_expected.takeError()));
    return;
  }
  EagerContext* eager_ctx = eager_ctx_expected.get();

  // Get device.
  Device* device = nullptr;
  Status s = eager_ctx->local_device_mgr()->LookupDevice(device_name, &device);
  if (!s.ok()) {
    // The device name can be invalid in certain cases. Use default CPU device.
    VLOG(1) << s.error_message() << " using default CPU device.";
  }

  // First we convert tensorflow::Tensor to RuntimeFallbackTensors.
  llvm::SmallVector<RuntimeFallbackTensor, 4> tfrt_tensor_args;
  tfrt_tensor_args.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    auto* av = args[i];
    auto tf_tensor = av->get<tensorflow::Tensor>();

    tfrt::TensorMetadata md = tfd::GetTensorMetadata(tf_tensor);
    OwnedTensorHandle tensor_handle{tensorflow::TensorHandle::CreateLocalHandle(
        std::move(tf_tensor), /*d=*/device, /*op_device=*/device, eager_ctx)};

    tfrt_tensor_args.push_back(
        RuntimeFallbackTensor(md.shape, md.dtype, std::move(tensor_handle)));
  }

  llvm::SmallVector<tfrt::Tensor*, 4> tfrt_tensor_arg_ptrs;
  tfrt_tensor_arg_ptrs.reserve(args.size());
  for (auto& tensor : tfrt_tensor_args) tfrt_tensor_arg_ptrs.push_back(&tensor);

  llvm::SmallVector<RCReference<tfrt::AsyncValue>, 4> tfrt_tensor_results;
  tfrt_tensor_results.resize(results.size());

  auto chain = RuntimeFallbackExecute(
      exec_ctx, op_name.str().c_str(), device_name.c_str(),
      tfrt_tensor_arg_ptrs, tfrt::OpAttrsRef(op_attrs), tfrt_tensor_results);

  if (op_chain) *op_chain = chain.CopyRef();

  // After coreruntime returns, we check if there is any error. Currently we
  // assume runtime fallback execution is always synchronous.
  DCHECK(chain.IsAvailable());
  if (chain.IsError()) {
    EmitErrorAndSetInResults(exec_ctx, chain.GetError(), results.values());
    return;
  }

  // Finally we convert the runtime fallback results, which are
  // RuntimeFallbackTensor, back to tensorflow::Tensor that is expected by the
  // BEF kernel.
  for (int i = 0; i < results.size(); ++i) {
    auto& runtime_fallback_tensor =
        tfrt_tensor_results[i]->get<RuntimeFallbackTensor>();
    const tensorflow::Tensor* tf_tensor = nullptr;
    tensorflow::Status s =
        runtime_fallback_tensor.GetTensorHandle()->Tensor(&tf_tensor);
    DCHECK(s.ok()) << s.ToString();
    results[i] =
        tfrt::MakeAvailableAsyncValueRef<tensorflow::Tensor>(*tf_tensor);
  }
}

Chain AddRuntimeFallbackImplicitConversionKernel(
    Argument<tfrt::OpHandler*> op_handler, const ExecutionContext& exec_ctx) {
  assert(op_handler.get()->GetName() == tfrt::CpuOpHandler::kName);
  tfrt::CpuOpHandler* cpu_op_handler =
      reinterpret_cast<tfrt::CpuOpHandler*>(op_handler.get());
  cpu_op_handler->AddImplicitConversion(RuntimeFallbackTensor::kTensorType,
                                        DenseHostTensor::kTensorType);
  cpu_op_handler->AddImplicitConversion(RuntimeFallbackTensor::kTensorType,
                                        tfrt::AnyScalarHostTensor::kTensorType);
  cpu_op_handler->AddImplicitConversion(RuntimeFallbackTensor::kTensorType,
                                        tfrt::StringHostTensor::kTensorType);
  return {};
}

void CreateRuntimeFallbackOpHandlerKernel(Result<tfrt::OpHandler*> op_handler,
                                          StringAttribute tf_device_name,
                                          const ExecutionContext& exec_ctx) {
  auto* runtime = tfrt::CoreRuntime::GetFromHostContext(exec_ctx.host());
  assert(runtime);
  auto op_handler_ptr =
      CreateRuntimeFallbackOpHandler(runtime, tf_device_name.get());
  assert(op_handler_ptr);
  op_handler.Emplace(op_handler_ptr.get());
}

static OwnedTensorHandle ConvertTFRTTensorToTFTensorHandle(
    tfrt::Tensor* tensor) {
  if (auto* dht = llvm::dyn_cast<tfrt::DenseHostTensor>(tensor)) {
    tensorflow::Tensor tensor =
        MoveHostBufferToTfTensor(dht->buffer(), dht->dtype(), dht->shape());

    return OwnedTensorHandle{
        tensorflow::TensorHandle::CreateLocalHandle(tensor)};
  }

  if (auto* sht = llvm::dyn_cast<tfrt::StringHostTensor>(tensor)) {
    tensorflow::Tensor tensor = CopyShtToTfTensor(*sht);
    return OwnedTensorHandle{
        tensorflow::TensorHandle::CreateLocalHandle(tensor)};
  }

  llvm_unreachable("unsupported tensor type");
}

static llvm::Expected<tfrt::Value> ConvertTFTensorHandleToTFRTTensor(
    OwnedTensorHandle tensor_handle, HostContext* host) {
  tensorflow::Status status;
  // Resolve ensures Tensor is on host CPU.
  OwnedAbstractTensorInterface tensor_interface{
      tensor_handle->Resolve(&status)};
  if (!status.ok()) {
    return tfrt::MakeStringError("error resolving TensorHandle: ",
                                 status.error_message());
  }
  auto tf_dtype = tensor_interface->Type();
  if (tf_dtype == DT_STRING) {
    // TODO(tfrt-devs): Consider a more efficient way to pass string
    // tensors between TFRT and TF.
    auto string_host_tensor =
        CopyTfStringTensorToStringHostTensor(tensor_interface.get(), host);
    if (!string_host_tensor)
      return tfrt::MakeStringError(
          "error converting TF string tensor to tfrt::StringHostTensor: ",
          string_host_tensor.takeError());
    return tfrt::Value(std::move(*string_host_tensor));
  }

  tfrt::TensorMetadata metadata(GetTfrtDtype(tf_dtype),
                                GetShape(tensor_interface.get()));

  CheckBoolCompatibility();
  void* data = tensor_interface->Data();
  size_t size = tensor_interface->ByteSize();
  // `tensor_interface` holds a reference on underlying Tensorflow buffer and is
  // held alive by HostBuffer deallocator lambda capture (a
  // llvm::unique_function), and it gets released when HostBuffer deallocator is
  // called and destroyed.
  auto host_buffer = HostBuffer::CreateFromExternal(
      data, size,
      [tensor_interface = std::move(tensor_interface)](void*, size_t) {});

  tfrt::Value value;
  value.emplace<DenseHostTensor>(metadata, std::move(host_buffer));
  return std::move(value);
}

void RegisterTfdDelegateKernels(tfrt::KernelRegistry* registry) {
  registry->AddKernel("tfd.init_eager_context",
                      TFRT_KERNEL(TfdInitEagerContext));
  registry->AddKernel("tfd.delegate_kernel",
                      TFRT_KERNEL(RuntimeFallbackKernel));
  registry->AddKernel("tfd.move_dht_to_tft", TFRT_KERNEL(TfdMoveDHTToTFT));
  registry->AddKernel("tfd.convert_tft_to_dht",
                      TFRT_KERNEL(TfdConvertTFTToDHT));
  registry->AddKernel("tfd.print_tft", TFRT_KERNEL(TfdPrintTFT));
  registry->AddKernel("tfrt_fallback_async.const_dense_tensor",
                      TFRT_KERNEL(ConstDenseTensor));
  registry->AddKernel("tfrt_fallback_async.const_string_tensor",
                      TFRT_KERNEL(ConstStringTensor));
  registry->AddKernel(
      "tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor",
      TFRT_KERNEL(CoreRTTensorHandleToFallbackTensor));
  registry->AddKernel(
      "tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle",
      TFRT_KERNEL(FallbackTensorToCoreRTTensorHandle));

  // TODO(b/187106271): Move fallback kernels to fallback only libraries so that
  // we don't have to depend on or link in corert kernels.
  registry->AddKernel("tfrt_fallback_async.predicate", TFRT_KERNEL(Predicate));
  registry->AddKernel("tfrt_fallback_async.print_tensor",
                      TFRT_KERNEL(PrintFallbackTensor));
  registry->AddKernel("corert.create_runtime_fallback_op_handler",
                      TFRT_KERNEL(CreateRuntimeFallbackOpHandlerKernel));
  registry->AddKernel("corert.add_runtime_fallback_implicit_conversions",
                      TFRT_KERNEL(AddRuntimeFallbackImplicitConversionKernel));
}

}  // namespace tfd
}  // namespace tensorflow
