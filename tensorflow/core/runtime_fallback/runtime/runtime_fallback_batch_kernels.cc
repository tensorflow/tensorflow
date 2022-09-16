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
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/execute_op_impl.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/attribute_utils.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor_view.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {
namespace {

using ::tfrt::AggregateAttr;
using ::tfrt::ArrayRef;
using ::tfrt::AsyncValue;
using ::tfrt::AsyncValueRef;
using ::tfrt::BEFAttributeType;
using ::tfrt::Chain;
using ::tfrt::CoreRuntimeOp;
using ::tfrt::DenseHostTensor;
using ::tfrt::MutableArrayRef;
using ::tfrt::MutableDHTArrayView;
using ::tfrt::RCReference;
using ::tfrt::string_view;

// Adapted from tfrt::ExecuteOpImpl, with additional logic to override OpAttrs.
void ExecuteFallbackOp(CoreRuntimeOp op,
                       MutableArrayRef<tfrt::TensorHandle> th_args,
                       AsyncValueRef<Chain>* op_chain,
                       MutableArrayRef<RCReference<AsyncValue>> results,
                       AggregateAttr op_attr_array,
                       RCReference<const tfrt::Function> f,
                       std::unique_ptr<tfrt::ExecutionContext> exec_ctx_ptr) {
  assert(!results.empty());

  llvm::SmallVector<tfrt::TensorHandle, 8> result_ths;
  result_ths.resize(results.size());

  tfrt::OpAttrs op_attrs;
  for (size_t i = 0, e = op_attr_array.GetNumElements(); i != e; ++i) {
    auto pair = op_attr_array.GetAttributeOfType<AggregateAttr>(i);
    assert(pair.GetNumElements() == 2);
    string_view key = pair.GetAttributeOfType<tfrt::StringAttr>(0).GetValue();
    // Make sure from lowering that we don't already have a "f" key.
    assert(key != "f");
    tfrt::TypedAttrBase attr = pair.GetAttribute(1);

    BEFAttributeType attribute_type = attr.type();
    if (IsArrayAttribute(attribute_type)) {
      auto type = GetOpAttrTypeFromBEFAttributeType(
          GetElementAttributeType(attribute_type));
      auto array_attr = attr.cast<tfrt::ArrayAttr>();
      op_attrs.SetRaw(key, array_attr.GetElements(), type,
                      array_attr.GetNumElements(),
                      tfrt::OpAttrsRawEntryType::kArray);
    } else if (IsDenseAttribute(attribute_type)) {
      auto r = op_attrs.Set(key, attr.cast<tfrt::DenseAttr>());
      assert(r);
      (void)r;
    } else if (IsDataTypeAttribute(attribute_type)) {
      switch (GetDataType(attribute_type)) {
        case tfrt::DType::I1:
          op_attrs.Set(key, attr.cast<tfrt::I1Attr>().GetValue());
          break;
        case tfrt::DType::I32:
          op_attrs.Set(key, attr.cast<tfrt::I32Attr>().GetValue());
          break;
        case tfrt::DType::I64:
          op_attrs.Set(key, attr.cast<tfrt::I64Attr>().GetValue());
          break;
        case tfrt::DType::F32:
          op_attrs.Set(key, attr.cast<tfrt::F32Attr>().GetValue());
          break;
        case tfrt::DType::F64:
          op_attrs.Set(key, attr.cast<tfrt::F64Attr>().GetValue());
          break;
        case tfrt::DType::String:
          op_attrs.SetString(key, attr.cast<tfrt::StringAttr>().GetValue());
          break;
        default:
          llvm_unreachable("unknown attribute type");
      }
    } else {
      switch (attribute_type) {
        case BEFAttributeType::kType: {
          auto type_attr = attr.cast<tfrt::TypeAttr>();
          tfrt::DType type = type_attr.GetValue();
          op_attrs.Set(key, tfrt::GetOpAttrTypeFromDType(type));
          break;
        }
        case BEFAttributeType::kShape:
          op_attrs.Set(key, attr.cast<tfrt::ShapeAttr>());
          break;
        case BEFAttributeType::kAggregate:
          op_attrs.Set(key, attr.cast<tfrt::AggregateAttr>());
          break;
        default:
          llvm_unreachable("unknown attribute type");
      }
    }
  }
  // Pass in a BEF function pointer with a I64 attribute.
  {
    int64_t ptr_value = reinterpret_cast<int64_t>(f.get());
    op_attrs.SetRaw("tfrt_bef_func", &ptr_value, tfrt::OpAttrType::I64,
                    /*element_count=*/1, tfrt::OpAttrsRawEntryType::kScalar);
  }

  op(*exec_ctx_ptr, th_args, tfrt::OpAttrsRef(op_attrs), result_ths, op_chain);

  AsyncWaitForResultsFromTensorHandles(results, result_ths);
}

void BatchFunctionFallback(tfrt::Argument<Chain> in_op_chain,
                           tfrt::RepeatedArguments<tfrt::TensorHandle> args,
                           tfrt::Result<Chain> out_op_chain,
                           tfrt::RemainingResults results,
                           AggregateAttr op_attr_array,
                           tfrt::Attribute<tfrt::Function> f,
                           tfrt::KernelErrorHandler handler,
                           const tfrt::ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  auto* runtime = tfrt::CoreRuntime::GetFromHostContext(host);
  assert(runtime);
  // TODO(b/161993570): Cleanup this magic string constant.
  constexpr tfrt::string_view kKernelFallbackOpHandlerName = "tfkernel";
  auto* op_handler = runtime->GetOpHandler(kKernelFallbackOpHandlerName);
  assert(op_handler && "fallback op_handler not found");

  constexpr char kTfKernelNameToFallback[] = "tf._BatchFunctionFallback";
  auto op = runtime->MakeOp(kTfKernelNameToFallback, op_handler);
  if (!op) return handler.ReportError(tfrt::StrCat(op.takeError()));

  llvm::SmallVector<tfrt::TensorHandle, 8> th_args;
  th_args.reserve(args.size() + 1);
  for (const auto& arg : args) {
    th_args.push_back(arg.CopyRef());
  }
  auto exec_ctx_ptr = std::make_unique<tfrt::ExecutionContext>(exec_ctx);
  // Pass in a ExecutionContext pointer as the last argument.
  {
    auto dht = tfrt::MakeAvailableAsyncValueRef<DenseHostTensor>(
        DenseHostTensor::CreateUninitialized<int64_t>(tfrt::TensorShape({}),
                                                      host)
            .getValue());
    auto view = MutableDHTArrayView<int64_t>(&*dht);
    *view.begin() = reinterpret_cast<int64_t>(exec_ctx_ptr.get());
    th_args.emplace_back(host->GetHostDeviceRef(), dht->metadata(),
                         std::move(dht));
  }

  llvm::SmallVector<RCReference<AsyncValue>, 8> results_refs;
  for (int b = 0, e = results.size(); b < e; ++b) {
    results_refs.push_back(results.AllocateAt<tfrt::TensorHandle>(b));
  }

  auto op_chain = in_op_chain.ValueRef();
  ExecuteFallbackOp(std::move(*op), th_args, &op_chain, results_refs,
                    op_attr_array, FormRef(&(*f)), std::move(exec_ctx_ptr));

  out_op_chain.Set(std::move(op_chain));
}

}  // namespace

void RegisterBatchFallbackKernels(tfrt::KernelRegistry* reg) {
  reg->AddKernel("tfrt_fallback_async.batch_function",
                 TFRT_KERNEL(BatchFunctionFallback));
}

}  // namespace tfd
}  // namespace tensorflow
