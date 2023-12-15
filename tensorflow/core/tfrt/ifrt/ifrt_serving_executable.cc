/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {
namespace {

absl::StatusOr<tensorflow::DataType> GetTensorDataType(
    xla::ifrt::DType ifrt_dtype) {
  if (ifrt_dtype.kind() == xla::ifrt::DType::kString) {
    return tensorflow::DataType::DT_STRING;
  }
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(ifrt_dtype));
  return tensorflow::EncodePrimitiveTypeAsDataType(primitive_type);
}
}  // namespace

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
IfrtServingExecutable::ConvertTensorToArray(const tensorflow::Tensor& tensor) {
  xla::PrimitiveType primitive_type;
  TF_RETURN_IF_ERROR(
      tensorflow::DataTypeToPrimitiveType(tensor.dtype(), &primitive_type));

  TF_ASSIGN_OR_RETURN(auto dtype, xla::ifrt::ToDType(primitive_type));

  xla::ifrt::Shape xla_shape(tensor.shape().dim_sizes());

  TF_ASSIGN_OR_RETURN(
      auto single_array,
      ifrt_client_->MakeArrayFromHostBuffer(
          tensor.data(), dtype, xla_shape,
          /*byte_strides=*/std::nullopt,
          xla::ifrt::SingleDeviceSharding::Create(ifrt_client_->devices()[0],
                                                  xla::ifrt::MemoryKind()),
          xla::ifrt::Client::HostBufferSemantics::kZeroCopy, [tensor]() {
            // keep a copy of tensor to keep it alive
          }));
  return single_array;
}

xla::ifrt::Future<absl::StatusOr<std::shared_ptr<xla::ifrt::LoadedExecutable>>>
IfrtServingExecutable::LookUpOrCreateExecutable(
    absl::Span<const tensorflow::Tensor> inputs) {
  std::vector<tensorflow::TensorShape> input_shapes;
  for (const auto& tensor : inputs) {
    input_shapes.push_back(tensor.shape());
  }
  Key key(input_shapes);

  xla::ifrt::Promise<
      absl::StatusOr<std::shared_ptr<xla::ifrt::LoadedExecutable>>>
      promise;
  xla::ifrt::Future<
      absl::StatusOr<std::shared_ptr<xla::ifrt::LoadedExecutable>>>
      future;

  {
    absl::MutexLock lock(&mutex_);

    const auto it = ifrt_executables_.find(key);
    if (it != ifrt_executables_.end()) {
      return it->second;
    }

    // Only create promise and future when cache missed.
    promise = xla::ifrt::Future<absl::StatusOr<
        std::shared_ptr<xla::ifrt::LoadedExecutable>>>::CreatePromise();
    future = xla::ifrt::Future<
        absl::StatusOr<std::shared_ptr<xla::ifrt::LoadedExecutable>>>(promise);

    ifrt_executables_.emplace(key, future);
  }

  LOG(INFO) << "Cache missed. Building executable";

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> mlir_hlo_module =
      CompileTfToHlo(*module_, inputs, signature_name(),
                     ifrt_client_->GetDefaultCompiler(),
                     shape_representation_fn_);
  if (!mlir_hlo_module.ok()) {
    promise.Set(mlir_hlo_module.status());
    return future;
  }

  absl::StatusOr<std::unique_ptr<xla::ifrt::LoadedExecutable>> ifrt_executable =
      ifrt_client_->GetDefaultCompiler()->Compile(
          std::make_unique<xla::ifrt::XlaProgram>(mlir_hlo_module->get()),
          std::make_unique<xla::ifrt::XlaCompileOptions>());
  if (!ifrt_executable.ok()) {
    promise.Set(ifrt_executable.status());
    return future;
  }

  promise.Set(std::shared_ptr<xla::ifrt::LoadedExecutable>(
      std::move(*ifrt_executable)));
  return future;
}

absl::StatusOr<std::vector<tensorflow::Tensor>> IfrtServingExecutable::Execute(
    absl::Span<const tensorflow::Tensor> inputs) {
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<xla::ifrt::LoadedExecutable> ifrt_executable,
      LookUpOrCreateExecutable(inputs).Await());

  std::vector<tsl::RCReference<xla::ifrt::Array>> args;
  args.reserve(inputs.size());
  for (auto& tensor : inputs) {
    TF_ASSIGN_OR_RETURN(auto single_array, ConvertTensorToArray(tensor));
    args.push_back(single_array);
  }

  TF_ASSIGN_OR_RETURN(auto execution_result,
                      ifrt_executable->Execute(
                          absl::MakeSpan(args),
                          /*options=*/{.untuple_result = true}, std::nullopt));

  auto status = execution_result.status.Await();
  TF_RETURN_IF_ERROR(status);

  std::vector<tensorflow::Tensor> outputs;
  std::vector<xla::ifrt::Future<absl::Status>> output_futures;
  output_futures.reserve(execution_result.outputs.size());
  outputs.reserve(execution_result.outputs.size());
  for (const auto& array_for_copy : execution_result.outputs) {
    tensorflow::TensorShape tensor_shape;
    TF_RETURN_IF_ERROR(tensorflow::TensorShape::BuildTensorShape(
        array_for_copy->shape().dims(), &tensor_shape));
    TF_ASSIGN_OR_RETURN(tensorflow::DataType data_type,
                        GetTensorDataType(array_for_copy->dtype()));

    tensorflow::Tensor tensor(data_type, std::move(tensor_shape));

    xla::ifrt::Future<absl::Status> copy_future =
        array_for_copy->CopyToHostBuffer(
            tensor.data(), /*byte_strides=*/std::nullopt,
            xla::ifrt::ArrayCopySemantics::kAlwaysCopy);

    output_futures.push_back(copy_future);
    outputs.push_back(std::move(tensor));
  }

  TF_RETURN_IF_ERROR(
      xla::ifrt::JoinFutures(absl::MakeSpan(output_futures)).Await());
  return outputs;
}

}  // namespace ifrt_serving
}  // namespace tensorflow
