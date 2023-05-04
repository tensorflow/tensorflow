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

#include "tensorflow/c/kernels.h"

#include <memory>
#include <vector>

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_buffer_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
// Required for IS_MOBILE_PLATFORM definition
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/tsl/framework/device_id_utils.h"
#include "tensorflow/tsl/platform/statusor.h"
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

using tensorflow::errors::InvalidArgument;
// This file forms the basis of a stable ABI for third-party kernel
// implementations. It is crucial that changes to this file are made cautiously
// and with a focus on maintaining both source and binary compatibility.

typedef std::function<void()> AsyncOpKernelDoneCallback;
void TF_RunAsyncOpKernelDoneCallback(TF_AsyncOpKernelDoneCallback* done) {
  (*reinterpret_cast<AsyncOpKernelDoneCallback*>(done))();
}

struct TF_KernelBuilder {
  ::tensorflow::KernelDefBuilder* cc_builder;

  void* (*create_function)(TF_OpKernelConstruction*);
  void (*compute_function)(void*, TF_OpKernelContext*);
  void (*compute_async_function)(void*, TF_OpKernelContext*,
                                 TF_AsyncOpKernelDoneCallback* done);
  void (*delete_function)(void*);
};

TF_KernelBuilder* TF_NewKernelBuilder(
    const char* op_name, const char* device_name,
    void* (*create_func)(TF_OpKernelConstruction*),
    void (*compute_func)(void*, TF_OpKernelContext*),
    void (*delete_func)(void*)) {
  TF_KernelBuilder* result = new TF_KernelBuilder;
  result->cc_builder = new ::tensorflow::KernelDefBuilder(op_name);
  result->cc_builder->Device(device_name);
  result->create_function = create_func;
  result->compute_function = compute_func;
  result->compute_async_function = nullptr;
  result->delete_function = delete_func;
  return result;
}

TF_KernelBuilder* TF_NewAsyncKernelBuilder(
    const char* op_name, const char* device_name,
    void* (*create_func)(TF_OpKernelConstruction*),
    void (*compute_async_func)(void*, TF_OpKernelContext*,
                               TF_AsyncOpKernelDoneCallback* done),
    void (*delete_func)(void*)) {
  TF_KernelBuilder* result = new TF_KernelBuilder;
  result->cc_builder = new ::tensorflow::KernelDefBuilder(op_name);
  result->cc_builder->Device(device_name);
  result->create_function = create_func;
  result->compute_function = nullptr;
  result->compute_async_function = compute_async_func;
  result->delete_function = delete_func;
  return result;
}

void TF_DeleteKernelBuilder(TF_KernelBuilder* builder) {
  if (builder != nullptr) {
    delete builder->cc_builder;
    delete builder;
  }
}

namespace tensorflow {
namespace {

#define CASE(type)                                               \
  case DataTypeToEnum<type>::value: {                            \
    kernel_builder->cc_builder->TypeConstraint<type>(attr_name); \
    break;                                                       \
  }

void AddTypeConstraint(TF_KernelBuilder* kernel_builder, const char* attr_name,
                       const DataType dtype, TF_Status* status) {
  // This needs to be under tensorflow:: namespace so that
  // TF_CALL_ALL_TYPES macro can find tensorflow::string as string.
  switch (dtype) {
    TF_CALL_ALL_TYPES(CASE);
    TF_CALL_QUANTIZED_TYPES(CASE);
    TF_CALL_quint16(CASE);
    TF_CALL_qint16(CASE);
    default:
      status->status = errors::Unimplemented("Unexpected type ", dtype);
      return;
  }
  TF_SetStatus(status, TF_OK, "");
}
#undef CASE

}  // namespace
}  // namespace tensorflow

namespace {
const tensorflow::AttrValue* GetAttrValue(TF_OpKernelConstruction* ctx,
                                          const char* attr_name,
                                          TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  const tensorflow::AttrValue* attr =
      ::tensorflow::AttrSlice(cc_ctx->def()).Find(attr_name);
  if (attr == nullptr) {
    status->status = InvalidArgument("Operation '", cc_ctx->def().name(),
                                     "' has no attr named '", attr_name, "'.");
  }
  return attr;
}
}  // namespace

void TF_KernelBuilder_TypeConstraint(TF_KernelBuilder* kernel_builder,
                                     const char* attr_name,
                                     const TF_DataType type,
                                     TF_Status* status) {
  tensorflow::DataType dtype = static_cast<tensorflow::DataType>(type);
  tensorflow::AddTypeConstraint(kernel_builder, attr_name, dtype, status);
}

void TF_KernelBuilder_HostMemory(TF_KernelBuilder* kernel_builder,
                                 const char* arg_name) {
  kernel_builder->cc_builder->HostMemory(arg_name);
}

void TF_KernelBuilder_Priority(TF_KernelBuilder* kernel_builder,
                               int32_t priority_number) {
  kernel_builder->cc_builder->Priority(priority_number);
}

void TF_KernelBuilder_Label(TF_KernelBuilder* kernel_builder,
                            const char* label) {
  kernel_builder->cc_builder->Label(label);
}

namespace tensorflow {
namespace {

// An OpKernel whose methods delegate to C function pointers.
class COpKernel : public OpKernel {
 public:
  explicit COpKernel(OpKernelConstruction* ctx,
                     void* (*create_func)(TF_OpKernelConstruction*),
                     void (*compute_func)(void*, TF_OpKernelContext*),
                     void (*delete_func)(void*))
      : OpKernel(ctx), compute_func_(compute_func), delete_func_(delete_func) {
    if (create_func != nullptr) {
      c_kernel_ =
          (*create_func)(reinterpret_cast<TF_OpKernelConstruction*>(ctx));
    } else {
      c_kernel_ = nullptr;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    (*compute_func_)(c_kernel_, reinterpret_cast<TF_OpKernelContext*>(ctx));
  }

  ~COpKernel() override {
    if (delete_func_ != nullptr) {
      (*delete_func_)(c_kernel_);
    }
  }

 private:
  void (*compute_func_)(void*, TF_OpKernelContext* context);
  void (*delete_func_)(void*);
  void* c_kernel_;
};

class CAsyncOpKernel : public AsyncOpKernel {
 public:
  explicit CAsyncOpKernel(
      OpKernelConstruction* ctx, void* (*create_func)(TF_OpKernelConstruction*),
      void (*compute_async_func)(void*, TF_OpKernelContext*,
                                 TF_AsyncOpKernelDoneCallback*),
      void (*delete_func)(void*))
      : AsyncOpKernel(ctx),
        compute_async_func_(compute_async_func),
        delete_func_(delete_func) {
    if (create_func != nullptr) {
      c_kernel_ =
          (*create_func)(reinterpret_cast<TF_OpKernelConstruction*>(ctx));
    } else {
      c_kernel_ = nullptr;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    Notification n;
    ComputeAsync(ctx, [&n]() { n.Notify(); });
    n.WaitForNotification();
  }

  void ComputeAsync(OpKernelContext* ctx, AsyncOpKernelDoneCallback done) {
    (*compute_async_func_)(
        c_kernel_, reinterpret_cast<TF_OpKernelContext*>(ctx),
        reinterpret_cast<TF_AsyncOpKernelDoneCallback*>(&done));
  }

  CAsyncOpKernel* AsAsync() override { return this; }

  ~CAsyncOpKernel() override {
    if (delete_func_ != nullptr) {
      (*delete_func_)(c_kernel_);
    }
  }

 private:
  void (*compute_async_func_)(void*, TF_OpKernelContext* context,
                              TF_AsyncOpKernelDoneCallback* done);
  void (*delete_func_)(void*);
  void* c_kernel_;
};

// A KernelFactory that returns COpKernel instances.
class KernelBuilderFactory
    : public ::tensorflow::kernel_factory::OpKernelFactory {
 public:
  explicit KernelBuilderFactory(TF_KernelBuilder* builder)
      : builder_(builder) {}
  ::tensorflow::OpKernel* Create(
      ::tensorflow::OpKernelConstruction* context) override {
    if (builder_->compute_function)
      return new ::tensorflow::COpKernel(context, builder_->create_function,
                                         builder_->compute_function,
                                         builder_->delete_function);
    else
      return new ::tensorflow::CAsyncOpKernel(
          context, builder_->create_function, builder_->compute_async_function,
          builder_->delete_function);
  }
  ~KernelBuilderFactory() override { TF_DeleteKernelBuilder(builder_); }

 private:
  TF_KernelBuilder* builder_;
};
}  // namespace
}  // namespace tensorflow

void TF_RegisterKernelBuilder(const char* name, TF_KernelBuilder* builder,
                              TF_Status* status) {
  using tensorflow::register_kernel::Name;

  TF_RegisterKernelBuilderWithKernelDef(
      /*serialized_kernel_def=*/nullptr, name, builder, status);
}

void TF_RegisterKernelBuilderWithKernelDef(const char* serialized_kernel_def,
                                           const char* name,
                                           TF_KernelBuilder* builder,
                                           TF_Status* status) {
  using tensorflow::register_kernel::Name;
  if (serialized_kernel_def == nullptr) {
    // If user doesn't provide a serialized KernelDef, use the kernel builder
    // to build a new one.
    tensorflow::kernel_factory::OpKernelRegistrar(
        builder->cc_builder->Build(), name,
        std::make_unique<tensorflow::KernelBuilderFactory>(builder));

    TF_SetStatus(status, TF_OK, "");
    return;
  }

  tensorflow::KernelDef* kernel_def = new tensorflow::KernelDef();
  bool success = kernel_def->ParsePartialFromString(serialized_kernel_def);
  if (!success) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Error parsing serialized KernelDef.");
    return;
  }

  tensorflow::kernel_factory::OpKernelRegistrar(
      kernel_def, name,
      std::make_unique<tensorflow::KernelBuilderFactory>(builder));

  TF_SetStatus(status, TF_OK, "");
}

// This function is only for pluggable device.
// It will return nullptr in all other cases.
// This function is experimental and subject to change.
SP_Stream TF_GetStream(TF_OpKernelContext* ctx, TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Accessing device stream is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
  return nullptr;
#else
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (cc_ctx->op_device_context() == nullptr) {  // CPU Device
    status->status = tensorflow::errors::FailedPrecondition(
        "Accessing device stream is not supported for a CPU device.");
    return nullptr;
  } else if (!cc_ctx->op_device_context()->IsPluggableDevice()) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Accessing device stream is only supported for pluggable devices.");
    return nullptr;
  } else {  // Is a PluggableDevice
    TF_SetStatus(status, TF_OK, "");
    auto c_stream = static_cast<stream_executor::CStream*>(
        cc_ctx->op_device_context()->stream()->implementation());
    return c_stream->Handle();
  }
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

int TF_NumInputs(TF_OpKernelContext* ctx) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  return cc_ctx->num_inputs();
}

int TF_NumOutputs(TF_OpKernelContext* ctx) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  return cc_ctx->num_outputs();
}

void TF_GetInput(TF_OpKernelContext* ctx, int i, TF_Tensor** tensor,
                 TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_inputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return;
  }
  const ::tensorflow::Tensor& cc_tensor(cc_ctx->input(i));
  if ((&cc_tensor) == nullptr) {
    *tensor = nullptr;
    return;
  }
  TF_Tensor* result =
      ::tensorflow::TF_TensorFromTensor(cc_tensor, &status->status);
  if (TF_GetCode(status) == TF_OK) {
    *tensor = result;
  }
}

void TF_InputRange(TF_OpKernelContext* ctx, const char* name,
                   TF_InputRange_Args* args) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  int start = -1, stop = -1;
  auto status = cc_ctx->op_kernel().InputRange(name, &start, &stop);
  args->start = start;
  args->stop = stop;
  tensorflow::Set_TF_Status_from_Status(args->status, status);
}

TF_DataType TF_InputDatatype(TF_OpKernelContext* ctx, int index) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  CHECK_GE(index, 0);                     // Crash OK
  CHECK_LT(index, cc_ctx->num_inputs());  // Crash OK
  return static_cast<TF_DataType>(cc_ctx->input_dtype(index));
}

void TF_SetOutput(TF_OpKernelContext* ctx, int i, const TF_Tensor* tensor,
                  TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_outputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "output index out of range");
    return;
  }
  ::tensorflow::Tensor cc_tensor;
  ::tensorflow::Status s = ::tensorflow::TF_TensorToTensor(tensor, &cc_tensor);
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(status, s);
  if (s.ok()) {
    cc_ctx->set_output(i, cc_tensor);
  }
}

TF_Tensor* TF_GetMutableOutput(TF_OpKernelContext* ctx, int i,
                               TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_outputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "output index out of range");
    return nullptr;
  }
  const ::tensorflow::Tensor& cc_tensor = *(cc_ctx->mutable_output(i));
  TF_Tensor* result =
      ::tensorflow::TF_TensorFromTensor(cc_tensor, &status->status);
  if (TF_GetCode(status) == TF_OK) {
    return result;
  } else {
    return nullptr;
  }
}

void TF_GetSerializedFunctionDefLibrary(
    TF_OpKernelContext* ctx, TF_Buffer* serialized_function_def_library,
    TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  auto fdef_lib =
      cc_ctx->function_library()->GetFunctionLibraryDefinition()->ToProto();
  auto cc_status =
      tensorflow::MessageToBuffer(fdef_lib, serialized_function_def_library);
  tensorflow::Set_TF_Status_from_Status(status, cc_status);
}

void TF_GetSerializedConfigProto(TF_OpKernelContext* ctx,
                                 TF_Buffer* serialized_config_proto,
                                 TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  const tensorflow::ConfigProto* config_proto_ptr =
      cc_ctx->function_library()->config_proto();
  tensorflow::ConfigProto config_proto;
  if (config_proto_ptr != nullptr) {
    config_proto = *config_proto_ptr;
  }
  auto cc_status =
      tensorflow::MessageToBuffer(config_proto, serialized_config_proto);
  tensorflow::Set_TF_Status_from_Status(status, cc_status);
}

void TF_GetSerializedResourceHandleProto(
    TF_OpKernelContext* ctx, int i, TF_Buffer* serialized_resource_handle_proto,
    TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  const tensorflow::ResourceHandle& handle = HandleFromInput(cc_ctx, i);
  tensorflow::ResourceHandleProto handle_proto;
  handle.AsProto(&handle_proto);
  auto cc_status = tensorflow::MessageToBuffer(
      handle_proto, serialized_resource_handle_proto);
  tensorflow::Set_TF_Status_from_Status(status, cc_status);
}

void TF_OpKernelConstruction_Failure(TF_OpKernelConstruction* ctx,
                                     TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s(::tensorflow::StatusFromTF_Status(status));
  cc_ctx->CtxFailure(s);
}

void TF_OpKernelContext_Failure(TF_OpKernelContext* ctx, TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  ::tensorflow::Status s(::tensorflow::StatusFromTF_Status(status));
  cc_ctx->CtxFailure(s);
}

void TF_OpKernelConstruction_GetAttrSize(TF_OpKernelConstruction* ctx,
                                         const char* attr_name,
                                         int32_t* list_size,
                                         int32_t* total_size,
                                         TF_Status* status) {
  const tensorflow::AttrValue* attr = GetAttrValue(ctx, attr_name, status);
  if (!status->status.ok()) {
    *list_size = -1;
    *total_size = -1;
    return;
  }
  switch (attr->value_case()) {
#define SINGLE_CASE(kK, attr_type, size_expr) \
  case tensorflow::AttrValue::kK:             \
    *list_size = -1;                          \
    *total_size = size_expr;                  \
    break;

    SINGLE_CASE(kS, TF_ATTR_STRING, attr->s().length());
    SINGLE_CASE(kI, TF_ATTR_INT, -1);
    SINGLE_CASE(kF, TF_ATTR_FLOAT, -1);
    SINGLE_CASE(kB, TF_ATTR_BOOL, -1);
    SINGLE_CASE(kType, TF_ATTR_TYPE, -1);
    SINGLE_CASE(kShape, TF_ATTR_SHAPE,
                attr->shape().unknown_rank() ? -1 : attr->shape().dim_size());
    SINGLE_CASE(kTensor, TF_ATTR_TENSOR, -1);
#undef SINGLE_CASE

    case tensorflow::AttrValue::kList:
      *list_size = 0;
      *total_size = -1;
#define LIST_CASE(field, attr_type, ...)      \
  if (attr->list().field##_size() > 0) {      \
    *list_size = attr->list().field##_size(); \
    __VA_ARGS__;                              \
    break;                                    \
  }

      LIST_CASE(
          s, TF_ATTR_STRING, *total_size = 0;
          for (int i = 0; i < attr->list().s_size();
               ++i) { *total_size += attr->list().s(i).size(); });
      LIST_CASE(i, TF_ATTR_INT);
      LIST_CASE(f, TF_ATTR_FLOAT);
      LIST_CASE(b, TF_ATTR_BOOL);
      LIST_CASE(type, TF_ATTR_TYPE);
      LIST_CASE(
          shape, TF_ATTR_SHAPE, *total_size = 0;
          for (int i = 0; i < attr->list().shape_size(); ++i) {
            const auto& s = attr->list().shape(i);
            *total_size += s.unknown_rank() ? 0 : s.dim_size();
          });
      LIST_CASE(tensor, TF_ATTR_TENSOR);
      LIST_CASE(tensor, TF_ATTR_FUNC);
#undef LIST_CASE
      break;

    case tensorflow::AttrValue::kPlaceholder:
      *list_size = -1;
      *total_size = -1;
      break;

    case tensorflow::AttrValue::kFunc:
      *list_size = -1;
      *total_size = -1;
      break;

    case tensorflow::AttrValue::VALUE_NOT_SET:
      status->status =
          InvalidArgument("Attribute '", attr_name, "' has no value set");
      break;
  }
}

#define DEFINE_TF_GETATTR(func, c_type, cc_type, attr_type, list_field)        \
  void TF_OpKernelConstruction_GetAttr##func(TF_OpKernelConstruction* ctx,     \
                                             const char* attr_name,            \
                                             c_type* val, TF_Status* status) { \
    TF_SetStatus(status, TF_OK, "");                                           \
    cc_type v;                                                                 \
    auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx); \
    ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &v);                   \
    ::tensorflow::Set_TF_Status_from_Status(status, s);                        \
    if (s.ok()) {                                                              \
      *val = static_cast<c_type>(v);                                           \
    }                                                                          \
  }                                                                            \
  void TF_OpKernelConstruction_GetAttr##func##List(                            \
      TF_OpKernelConstruction* ctx, const char* attr_name, c_type* vals,       \
      int max_vals, TF_Status* status) {                                       \
    TF_SetStatus(status, TF_OK, "");                                           \
    const tensorflow::AttrValue* attr = GetAttrValue(ctx, attr_name, status);  \
    if (!status->status.ok()) return;                                          \
    if (attr->value_case() != tensorflow::AttrValue::kList) {                  \
      status->status =                                                         \
          InvalidArgument("Value for '", attr_name, "' is not a list.");       \
      return;                                                                  \
    }                                                                          \
    status->status =                                                           \
        tensorflow::AttrValueHasType(*attr, "list(" attr_type ")");            \
    if (!status->status.ok()) return;                                          \
    const auto len = std::min(max_vals, attr->list().list_field##_size());     \
    for (int i = 0; i < len; ++i) {                                            \
      vals[i] = static_cast<c_type>(attr->list().list_field(i));               \
    }                                                                          \
  }

DEFINE_TF_GETATTR(Type, TF_DataType, tensorflow::DataType, "type", type)
DEFINE_TF_GETATTR(Int32, int32_t, int32_t, "int", i)
DEFINE_TF_GETATTR(Int64, int64_t, int64_t, "int", i)
DEFINE_TF_GETATTR(Float, float, float, "float", f)
DEFINE_TF_GETATTR(Bool, TF_Bool, bool, "bool", b)

void TF_OpKernelConstruction_GetAttrString(TF_OpKernelConstruction* ctx,
                                           const char* attr_name, char* value,
                                           size_t max_length,
                                           TF_Status* status) {
  std::string v;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &v);
  ::tensorflow::Set_TF_Status_from_Status(status, s);

  if (!status->status.ok()) return;

  if (max_length <= 0) {
    return;
  }
  std::memcpy(value, v.data(), std::min<size_t>(v.length(), max_length));
}

void TF_OpKernelConstruction_GetAttrStringList(TF_OpKernelConstruction* ctx,
                                               const char* attr_name,
                                               char** values, size_t* lengths,
                                               int max_values, void* storage,
                                               size_t storage_size,
                                               TF_Status* status) {
  std::vector<std::string> v;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &v);
  ::tensorflow::Set_TF_Status_from_Status(status, s);

  if (!status->status.ok()) return;

  const auto len = std::min(max_values, static_cast<int>(v.size()));
  char* p = static_cast<char*>(storage);
  for (int i = 0; i < len; ++i) {
    const std::string& s = v[i];
    values[i] = p;
    lengths[i] = s.size();
    if ((p + s.size()) > (static_cast<char*>(storage) + storage_size)) {
      status->status = InvalidArgument(
          "Not enough storage to hold the requested list of strings");
      return;
    }
    memcpy(values[i], s.data(), s.size());
    p += s.size();
  }
}

void TF_OpKernelConstruction_GetAttrTensor(TF_OpKernelConstruction* ctx,
                                           const char* attr_name,
                                           TF_Tensor** val, TF_Status* status) {
  *val = nullptr;
  ::tensorflow::Tensor t;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &t);
  ::tensorflow::Set_TF_Status_from_Status(status, s);

  if (!status->status.ok()) return;

  *val = TF_TensorFromTensor(t, &status->status);
}

void TF_OpKernelConstruction_GetAttrTensorList(TF_OpKernelConstruction* ctx,
                                               const char* attr_name,
                                               TF_Tensor** vals, int max_values,
                                               TF_Status* status) {
  std::vector<::tensorflow::Tensor> v;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &v);
  ::tensorflow::Set_TF_Status_from_Status(status, s);

  if (!status->status.ok()) return;

  const auto len = std::min(max_values, static_cast<int>(v.size()));
  for (int i = 0; i < len; ++i) {
    vals[i] = TF_TensorFromTensor(v[i], &status->status);
    if (!status->status.ok()) return;
  }
}

TF_Buffer* TF_OpKernelConstruction_GetAttrFunction(TF_OpKernelConstruction* ctx,
                                                   const char* attr_name,
                                                   TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  tensorflow::NameAttrList function;
  auto cc_status = cc_ctx->GetAttr(attr_name, &function);
  if (!cc_status.ok()) {
    tsl::Set_TF_Status_from_Status(status, cc_status);
    return nullptr;
  }
  TF_Buffer* buffer = TF_NewBuffer();
  cc_status = tensorflow::MessageToBuffer(function, buffer);
  tsl::Set_TF_Status_from_Status(status, cc_status);
  if (!cc_status.ok())
    return nullptr;
  else
    return buffer;
}

bool TF_OpKernelConstruction_HasAttr(TF_OpKernelConstruction* ctx,
                                     const char* attr_name, TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  return cc_ctx->HasAttr(attr_name);
}

TF_StringView TF_OpKernelConstruction_GetName(TF_OpKernelConstruction* ctx) {
  auto* cc_ctx = reinterpret_cast<tensorflow::OpKernelConstruction*>(ctx);
  TF_StringView string_view_of_name;
  string_view_of_name.data = cc_ctx->def().name().data();
  string_view_of_name.len = cc_ctx->def().name().length();
  return string_view_of_name;
}

TF_DataType TF_ExpectedOutputDataType(TF_OpKernelContext* ctx, int i) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  CHECK_GE(i, 0);
  CHECK_LT(i, cc_ctx->num_outputs());
  return static_cast<TF_DataType>(cc_ctx->expected_output_dtype(i));
}

bool TF_IsHostMemoryInput(TF_OpKernelContext* ctx, int i, TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_inputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return false;
  }
  TF_SetStatus(status, TF_OK, "");
  return cc_ctx->input_memory_type(i) == tensorflow::HOST_MEMORY;
}

bool TF_IsHostMemoryOutput(TF_OpKernelContext* ctx, int i, TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_outputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "output index out of range");
    return false;
  }
  TF_SetStatus(status, TF_OK, "");
  return cc_ctx->output_memory_type(i) == tensorflow::HOST_MEMORY;
}

int64_t TF_StepId(TF_OpKernelContext* ctx) {
  return reinterpret_cast<::tensorflow::OpKernelContext*>(ctx)->step_id();
}

TF_Buffer* TF_OpKernelConstruction_GetNodeDef(TF_OpKernelConstruction* ctx,
                                              TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  TF_Buffer* ret = TF_NewBuffer();
  status->status = MessageToBuffer(cc_ctx->def(), ret);
  if (!status->status.ok()) {
    TF_DeleteBuffer(ret);
    return nullptr;
  }
  return ret;
}

uint64_t TF_GetFrameId(TF_OpKernelContext* ctx) {
  return reinterpret_cast<::tensorflow::OpKernelContext*>(ctx)
      ->frame_iter()
      .frame_id;
}

int TF_GetGraphDefVersion(TF_OpKernelContext* ctx) {
  return reinterpret_cast<::tensorflow::OpKernelContext*>(ctx)
      ->function_library()
      ->graph_def_version();
}

int64_t TF_GetIterId(TF_OpKernelContext* ctx) {
  return reinterpret_cast<::tensorflow::OpKernelContext*>(ctx)
      ->frame_iter()
      .iter_id;
}

int64_t TF_GetStepId(TF_OpKernelContext* ctx) {
  return reinterpret_cast<::tensorflow::OpKernelContext*>(ctx)->step_id();
}

int TF_GetDeviceId(TF_OpKernelContext* ctx) {
  // TensorFlow always sets device in OpKernelContext.
  const tensorflow::DeviceBase* device_base =
      reinterpret_cast<tensorflow::OpKernelContext*>(ctx)->device();
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  if (!device_base->parsed_name().has_id) return -1;
  return device_base->parsed_name().id;
#else
  const auto* device = reinterpret_cast<const tensorflow::Device*>(
      device_base->UnderlyingDevice());
  const tsl::StatusOr<int> id = tsl::GetDeviceIdFromDeviceParsedName(
      device->parsed_name(), tensorflow::DeviceType(device->device_type()));
  if (!id.ok()) return -1;
  return *id;
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

TF_StringView TF_GetOpKernelName(TF_OpKernelContext* ctx) {
  auto cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  TF_StringView opkernel_name_sv;
  opkernel_name_sv.data = cc_ctx->op_kernel().name().data();
  opkernel_name_sv.len = cc_ctx->op_kernel().name().length();
  return opkernel_name_sv;
}

TF_StringView TF_GetResourceMgrDefaultContainerName(TF_OpKernelContext* ctx) {
  auto cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  TF_StringView default_container_name_sv;
  default_container_name_sv.data =
      cc_ctx->resource_manager()->default_container().data();
  default_container_name_sv.len =
      cc_ctx->resource_manager()->default_container().length();
  return default_container_name_sv;
}

TF_StringView TF_GetOpKernelRequestedInput(TF_OpKernelContext* ctx,
                                           size_t index) {
  auto cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  TF_StringView requested_input_sv;
  requested_input_sv.data = cc_ctx->op_kernel().requested_input(index).data();
  requested_input_sv.len = cc_ctx->op_kernel().requested_input(index).length();
  return requested_input_sv;
}

TF_Tensor* TF_AllocateOutput(TF_OpKernelContext* context, int index,
                             TF_DataType dtype, const int64_t* dims,
                             int num_dims, size_t len, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(context);
  tensorflow::gtl::ArraySlice<const int64_t> dimarray(
      reinterpret_cast<const int64_t*>(dims), num_dims);
  tensorflow::Tensor* tensor;
  tensorflow::Status s = cc_ctx->allocate_output(
      index, tensorflow::TensorShape(dimarray), &tensor);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  TF_Tensor* tf_tensor = TF_TensorFromTensor(*tensor, &s);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  return tf_tensor;
}

TF_Tensor* TF_ForwardInputOrAllocateOutput(
    TF_OpKernelContext* context, const int* candidate_input_indices,
    int num_candidate_input_indices, int output_index,
    const int64_t* output_dims, int output_num_dims, int* forwarded_input,
    TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(context);

  tensorflow::gtl::ArraySlice<int> input_indices_array(
      candidate_input_indices, num_candidate_input_indices);
  tensorflow::gtl::ArraySlice<const int64_t> output_dimarray(
      reinterpret_cast<const int64_t*>(output_dims), output_num_dims);
  tensorflow::Tensor* output_tensor_pointer;
  tensorflow::Status s = cc_ctx->forward_input_or_allocate_output(
      input_indices_array, output_index,
      tensorflow::TensorShape(output_dimarray), &output_tensor_pointer,
      forwarded_input);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  TF_Tensor* tf_tensor_output = TF_TensorFromTensor(*output_tensor_pointer, &s);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  return tf_tensor_output;
}

TF_Tensor* TF_AllocateTemp(TF_OpKernelContext* context, TF_DataType dtype,
                           const int64_t* dims, int num_dims,
                           TF_AllocatorAttributes* attributes,
                           TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(context);
  TF_SetStatus(status, TF_OK, "");
  tensorflow::gtl::ArraySlice<const int64_t> dimarray(
      reinterpret_cast<const int64_t*>(dims), num_dims);
  if (attributes && !attributes->struct_size) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        "TF_AllocatorAttributes struct "
        "size member must be set to TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE");
    return nullptr;
  }
  tensorflow::AllocatorAttributes allocator_attr;
  if (attributes && attributes->on_host) {
    allocator_attr.set_on_host(true);
  }
  tensorflow::Status s;
  tensorflow::Tensor tensor;
  s = cc_ctx->allocate_temp(static_cast<tensorflow::DataType>(dtype),
                            tensorflow::TensorShape(dimarray), &tensor,
                            allocator_attr);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  TF_Tensor* tf_tensor;
  tf_tensor = TF_TensorFromTensor(tensor, &s);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  return tf_tensor;
}
