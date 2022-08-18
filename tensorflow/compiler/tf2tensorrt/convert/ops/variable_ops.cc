/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferRuntimeCommon.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

class ConvertVariableV2 : public OpConverterBase<ConvertVariableV2> {
 public:
  ConvertVariableV2(OpConverterParams* params)
      : OpConverterBase<ConvertVariableV2>(params) {}

  struct VariableV2Attributes {
    TensorShapeProto shape_proto;
    TensorShape shape;
    string name;
    DataType dtype;
    string shared_name;
    string container;
  };

  static constexpr std::array<InputArgSpec, 0> InputSpec() { return {}; }

  static constexpr std::array<DataType, 2> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  static constexpr const char* NodeDefDataTypeAttributeName() {
    return "dtype";
  }

  template <typename T>
  Status ValidateImpl() {
    const auto& node_def = params_->node_def;

    // Verify and consume node attributes.
    StatusOr<TensorShapeProto> shape_proto =
        GetAttrValue<TensorShapeProto>("shape");
    StatusOr<string> shared_name = GetAttrValue<string>("shared_name");
    StatusOr<string> container = GetAttrValue<string>("container");
    TRT_ENSURE_OK(shape_proto);
    TRT_ENSURE_OK(shared_name);
    TRT_ENSURE_OK(container);

    attrs_.shape_proto = *shape_proto;
    attrs_.shape = TensorShape(*shape_proto);
    attrs_.name = node_def.name();
    attrs_.shared_name = *shared_name;
    attrs_.container = *container;

    Tensor tensor(attrs_.dtype, attrs_.shape);
    auto tensor_flat = tensor.flat<T>();
    for (int64_t i = 0; i < tensor_flat.size(); i++) {
      tensor_flat(i) = T(0.0f);
    }

    TRT_ShapedWeights weights;
    TF_RETURN_IF_ERROR(
        TfTensorToTrtWeights(tensor, params_->weight_store, &weights));

    // Only push outputs during validation and when outputs are expected.
    if (params_->validation_only && params_->outputs != nullptr) {
      AddOutput(TRT_TensorOrWeights(weights));
    }
    return Status::OK();
  }

  Status Validate() {
    const auto& node_def = params_->node_def;
    StatusOr<DataType> dtype = GetAttrValue<DataType>("dtype");
    TRT_ENSURE_OK(dtype);
    attrs_.dtype = *dtype;

    switch (attrs_.dtype) {
      case DT_FLOAT:
        return ValidateImpl<float>();
      case DT_HALF:
        return ValidateImpl<Eigen::half>();
      default:
        // Note: this should have been caught by ValidateNodeDefDataType, but
        // the compiler expects that all paths be handled in switch.
        return errors::Unimplemented("Data type ", DataTypeString(attrs_.dtype),
                                     " is not supported for ", node_def.op(),
                                     ", at ", node_def.name());
    }
  }

  template <typename T>
  Status ConvertImpl() {
    Tensor tensor(attrs_.dtype, attrs_.shape);
    auto tensor_flat = tensor.flat<T>();

    auto ctx = params_->converter->context();
    TRT_ENSURE(ctx != nullptr);
    auto lib = ctx->function_library();

    // Clone function library runtime in order to get a mutable library
    // definition to add and run a function with the variable operation.
    std::unique_ptr<FunctionLibraryDefinition> lib_def;
    std::unique_ptr<ProcessFunctionLibraryRuntime> lib_pflr;
    FunctionLibraryRuntime* lib_clone;  // Not owned.
    TF_RETURN_IF_ERROR(lib->Clone(&lib_def, &lib_pflr, &lib_clone));

    // Create function definition.
    string func_name = attrs_.name + "/func";
    FunctionDef fdef = FunctionDefHelper::Define(
        func_name,                                              // Name
        {},                                                     // Args
        {absl::StrCat("out: ", DataTypeString(attrs_.dtype))},  // Returns
        {},                                                     // Attr def
        // Nodes
        {{{attrs_.name},
          "VariableV2",
          {},
          {{"dtype", attrs_.dtype},
           {"shape", attrs_.shape_proto},
           {"container", attrs_.container},
           {"shared_name", attrs_.shared_name}}},
         {{"out"}, "Identity", {attrs_.name}, {{"T", attrs_.dtype}}}});

    // Add function definition to the library.
    TF_RETURN_IF_ERROR(lib_def->AddFunctionDef(fdef));

    // Instantiate function.
    FunctionLibraryRuntime::Handle func_handle;
    FunctionLibraryRuntime::InstantiateOptions inst_ops;
    inst_ops.state_handle = "";
    inst_ops.target = ctx->device()->name();
    AttrValueMap attr_list;
    TF_RETURN_IF_ERROR(lib_clone->Instantiate(func_name, AttrSlice(&attr_list),
                                              inst_ops, &func_handle));

    FunctionLibraryRuntime::Options opts;
    opts.rendezvous = ctx->rendezvous();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.runner = ctx->runner();

    std::vector<Tensor> args;  // empty
    std::vector<Tensor>* rets = new std::vector<Tensor>();
    std::unique_ptr<std::vector<Tensor>> outputs_wrapper(rets);

    // Run the new function synchronously.
    TF_RETURN_IF_ERROR(lib_clone->RunSync(opts, func_handle, args, rets));

    TRT_ENSURE(ctx->op_device_context() != nullptr);
    TRT_ENSURE(ctx->op_device_context()->stream() != nullptr);

    // Copy tensor.
    const cudaStream_t* stream =
        reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                  ->stream()
                                                  ->implementation()
                                                  ->GpuStreamMemberHack());
    TRT_ENSURE(stream != nullptr);

    auto ret = cudaMemcpyAsync(tensor_flat.data(), rets->at(0).flat<T>().data(),
                               rets->at(0).NumElements() * sizeof(T),
                               cudaMemcpyDeviceToHost, *stream);
    if (ret != 0) {
      return errors::Internal("Could not copy the variable ", attrs_.name);
    }
    cudaStreamSynchronize(*stream);

    TRT_ShapedWeights weights;
    TF_RETURN_IF_ERROR(
        TfTensorToTrtWeights(tensor, params_->weight_store, &weights));

    AddOutput(TRT_TensorOrWeights(weights));
    return Status::OK();
  }

  Status Convert() {
    const auto& node_def = params_->node_def;

    switch (attrs_.dtype) {
      case DT_FLOAT:
        return ConvertImpl<float>();
      case DT_HALF:
        return ConvertImpl<Eigen::half>();
      default:
        // Note: this should have been caught by ValidateNodeDefDataType, but
        // the compiler expects that all paths be handled in switch.
        return errors::Unimplemented("Data type ", DataTypeString(attrs_.dtype),
                                     " is not supported for ", node_def.op(),
                                     ", at ", node_def.name());
    }
  }

 private:
  VariableV2Attributes attrs_{};
};
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertVariableV2>(),
                                  {"VariableV2"});

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
