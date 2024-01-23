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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_tensor_utils.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

absl::StatusOr<xla::DeviceAssignment> GetXlaDeviceAssignment(
    const xla::ifrt::Client& ifrt_client,
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata) {
  int num_replicas = compile_metadata.num_replicas();
  int num_partitions = compile_metadata.num_cores_per_replica();

  VLOG(2) << " Number of replcas is " << num_replicas
          << " and num_partitions is " << num_partitions;

  if (num_replicas > 1) {
    return absl::UnimplementedError(
        absl::StrCat("Only support single replica, but replica number is ",
                     num_replicas, " and num_partitions is ", num_partitions));
  }

  if (compile_metadata.has_device_assignment()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::DeviceAssignment> da,
                        xla::DeviceAssignment::Deserialize(
                            compile_metadata.device_assignment()));

    return *std::move(da);
  } else {
    // TODO(b/316068010): integrate core selection.
    return ifrt_client.GetDefaultDeviceAssignment(num_replicas, num_partitions);
  }
}

absl::StatusOr<std::vector<xla::ifrt::Device*>> GetAssignedDevices(
    const xla::ifrt::Client& ifrt_client,
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata) {
  TF_ASSIGN_OR_RETURN(auto device_assignment,
                      GetXlaDeviceAssignment(ifrt_client, compile_metadata));

  const int num_devices =
      device_assignment.replica_count() * device_assignment.computation_count();
  std::vector<xla::ifrt::Device*> devices;
  devices.reserve(num_devices);
  for (int replica_idx = 0; replica_idx < device_assignment.replica_count();
       replica_idx++) {
    for (int computation_idx = 0;
         computation_idx < device_assignment.computation_count();
         computation_idx++) {
      auto device_id = device_assignment(replica_idx, computation_idx);
      TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                          ifrt_client.LookupDevice(device_id));
      devices.push_back(device);
    }
  }
  return devices;
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
CreateArrayFromHostTensorForSingleDevice(xla::ifrt::Client& ifrt_client,
                                         const tensorflow::Tensor& tensor,
                                         xla::ifrt::Device* device) {
  TF_ASSIGN_OR_RETURN(auto dtype, ToIfrtDType(tensor.dtype()));

  VLOG(2) << "Make single device array for buffer slice "
          << " at " << tensor.data();
  auto single_device_sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  return ifrt_client.MakeArrayFromHostBuffer(
      tensor.data(), dtype, ToIfrtShape(tensor.shape()),
      /*byte_strides=*/{}, std::move(single_device_sharding),
      xla::ifrt::Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
      [tensor]() {
        // Keep tensor alive
        VLOG(2) << "Done with single device host buffer for slice "
                << " at " << tensor.data();
      });
}

}  // namespace

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
IfrtServingExecutable::ConvertTensorToArray(
    const tensorflow::Tensor& tensor, const xla::ifrt::DeviceList& device_list,
    const xla::OpSharding& sharding) {
  xla::ifrt::Shape input_shape = ToIfrtShape(tensor.shape());
  VLOG(2) << "Converting tensor of shape " << input_shape;

  TF_ASSIGN_OR_RETURN(auto hlo_sharding, xla::HloSharding::FromProto(sharding));
  VLOG(3) << "IsTiled: " << hlo_sharding.IsTiled();
  VLOG(3) << "IsReplicated: " << hlo_sharding.IsReplicated();
  VLOG(3) << "IsTileMaximal: " << hlo_sharding.IsTileMaximal();
  if (!hlo_sharding.IsTiled() && !hlo_sharding.IsReplicated() &&
      !hlo_sharding.IsTileMaximal()) {
    return absl::UnimplementedError(absl::StrCat(
        "Only support MAXIMAL, OTHER or REPLICATED, but got sharding : ",
        hlo_sharding.ToProto()));
  }

  VLOG(1) << "Hlo sharding: " << hlo_sharding.ToString();
  VLOG(1) << "Device list size: " << device_list.size();

  if (device_list.size() == 1) {
    if (hlo_sharding.IsTiled()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Tiled sharding", hlo_sharding.ToString(),
                       " expect more than 1 device, but got 1 only"));
    }
    return CreateArrayFromHostTensorForSingleDevice(*ifrt_client_, tensor,
                                                    device_list[0]);
  }

  // Replicate implies Maximal, but not vice versa. Only Maximal is single
  // device.
  if (!hlo_sharding.IsReplicated() && hlo_sharding.IsTileMaximal()) {
    VLOG(1) << "Single device fast path for Maximal tiled tensor";
    xla::ifrt::Device* device;
    if (hlo_sharding.HasUniqueDevice()) {
      int unique_device_id = hlo_sharding.GetUniqueDevice();
      TF_ASSIGN_OR_RETURN(device, ifrt_client_->LookupDevice(unique_device_id));
    } else {
      device = device_list[0];
    }
    return CreateArrayFromHostTensorForSingleDevice(*ifrt_client_, tensor,
                                                    device);
  }

  return MakeAssembledArrayFromHostBuffer(*ifrt_client_, tensor,
                                          std::move(hlo_sharding), device_list,
                                          thread_pool_device_);
}

absl::StatusOr<IfrtServingExecutable::CachedExecutableBundle>
IfrtServingExecutable::CreateExecutableSynchronously(
    absl::Span<const tensorflow::Tensor> inputs) {
  TF_ASSIGN_OR_RETURN(Tf2HloResult tf2hlo_result,
                      CompileTfToHlo(*module_, inputs, signature_name(),
                                     *ifrt_client_, shape_representation_fn_));

  const int num_replicas = tf2hlo_result.compile_metadata.num_replicas();
  const int num_partitions =
      tf2hlo_result.compile_metadata.num_cores_per_replica();

  VLOG(2) << " Number of replcas is " << num_replicas
          << " and num_partitions is " << num_partitions;

  if (num_replicas > 1) {
    return absl::UnimplementedError(
        absl::StrCat("Only support single replica, but replica number is ",
                     num_replicas, " and num_partitions is ", num_partitions));
  }

  TF_ASSIGN_OR_RETURN(
      xla::DeviceAssignment da,
      GetXlaDeviceAssignment(*ifrt_client_, tf2hlo_result.compile_metadata));

  VLOG(2) << "Device assignment :" << da.ToString();

  xla::CompileOptions xla_compile_options;
  // TODO(b/304839793): populate xla_compile_options.argument_layouts.
  // TODO(b/316071625): per model config in TFRT + IFRT.
  xla_compile_options.executable_build_options.set_num_replicas(num_replicas);
  xla_compile_options.executable_build_options.set_num_partitions(
      num_partitions);

  xla_compile_options.executable_build_options.set_use_spmd_partitioning(true);
  xla_compile_options.parameter_is_tupled_arguments = false;
  xla_compile_options.executable_build_options.set_device_assignment(da);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::ifrt::LoadedExecutable> ifrt_executable,
      ifrt_client_->GetDefaultCompiler()->Compile(
          std::make_unique<xla::ifrt::XlaProgram>(
              tf2hlo_result.mlir_hlo_module.get()),
          std::make_unique<xla::ifrt::XlaCompileOptions>(xla_compile_options)));

  CachedExecutableBundle executable_bundle;
  executable_bundle.ifrt_executable = std::move(ifrt_executable);
  executable_bundle.compile_metadata =
      std::move(tf2hlo_result.compile_metadata);

  return executable_bundle;
}

xla::ifrt::Future<absl::StatusOr<IfrtServingExecutable::CachedExecutableBundle>>
IfrtServingExecutable::LookUpOrCreateExecutable(
    absl::Span<const tensorflow::Tensor> inputs) {
  std::vector<tensorflow::TensorShape> input_shapes;
  for (const auto& tensor : inputs) {
    input_shapes.push_back(tensor.shape());
  }
  Key key(input_shapes);

  xla::ifrt::Promise<absl::StatusOr<CachedExecutableBundle>> promise;
  xla::ifrt::Future<absl::StatusOr<CachedExecutableBundle>> future;

  {
    absl::MutexLock lock(&mutex_);

    const auto it = executable_bundles_.find(key);
    if (it != executable_bundles_.end()) {
      return it->second;
    }

    // Only create promise and future when cache missed.
    promise = xla::ifrt::Future<
        absl::StatusOr<CachedExecutableBundle>>::CreatePromise();
    future = xla::ifrt::Future<absl::StatusOr<CachedExecutableBundle>>(promise);

    executable_bundles_.emplace(key, future);
  }

  LOG(INFO) << "Cache missed. Building executable";
  absl::StatusOr<CachedExecutableBundle> executable_bundle =
      CreateExecutableSynchronously(inputs);

  promise.Set(std::move(executable_bundle));
  return future;
}

absl::StatusOr<std::vector<tensorflow::Tensor>> IfrtServingExecutable::Execute(
    absl::Span<const tensorflow::Tensor> inputs) {
  TF_ASSIGN_OR_RETURN(CachedExecutableBundle executable_bundle,
                      LookUpOrCreateExecutable(inputs).Await());

  std::vector<tsl::RCReference<xla::ifrt::Array>> args;
  args.reserve(inputs.size());

  TF_ASSIGN_OR_RETURN(
      std::vector<xla ::ifrt::Device*> devices,
      GetAssignedDevices(*ifrt_client_, executable_bundle.compile_metadata));
  xla::ifrt::DeviceList device_list(
      xla::ifrt::DeviceList::Devices(devices.begin(), devices.end()));

  auto compile_metadata_arg_iter =
      executable_bundle.compile_metadata.args().begin();
  if (executable_bundle.compile_metadata.args().size() != inputs.size()) {
    return absl::InternalError(absl::StrCat(
        "Expect ", executable_bundle.compile_metadata.args().size(),
        " but got ", inputs.size(), " arguments"));
  }
  for (const auto& input_tensor : inputs) {
    TF_ASSIGN_OR_RETURN(
        auto single_array,
        ConvertTensorToArray(input_tensor, device_list,
                             compile_metadata_arg_iter->sharding()));
    args.push_back(single_array);
    compile_metadata_arg_iter++;
  }

  VLOG(2) << "Start Execution";

  TF_ASSIGN_OR_RETURN(auto execution_result,
                      executable_bundle.ifrt_executable->Execute(
                          absl::MakeSpan(args),
                          /*options=*/{.untuple_result = true}, std::nullopt));

  auto status = execution_result.status.Await();
  TF_RETURN_IF_ERROR(status);

  std::vector<tensorflow::Tensor> outputs;
  std::vector<xla::ifrt::Future<absl::Status>> output_futures;
  output_futures.reserve(execution_result.outputs.size());
  outputs.reserve(execution_result.outputs.size());

  if (executable_bundle.compile_metadata.retvals().size() !=
      execution_result.outputs.size()) {
    return absl::InternalError(absl::StrCat(
        "Expect ", executable_bundle.compile_metadata.retvals().size(),
        " but got ", execution_result.outputs.size(), " outputs"));
  }
  for (int i = 0; i < execution_result.outputs.size(); ++i) {
    tensorflow::TensorShape tensor_shape;
    const tsl::RCReference<xla::ifrt::Array>& array_for_copy =
        execution_result.outputs[i];
    const tpu::TPUCompileMetadataProto::Retval& metadata_retval =
        executable_bundle.compile_metadata.retvals()[i];

    TF_RETURN_IF_ERROR(tensorflow::TensorShape::BuildTensorShape(
        array_for_copy->shape().dims(), &tensor_shape));
    TF_ASSIGN_OR_RETURN(tensorflow::DataType data_type,
                        ToTensorDataType(array_for_copy->dtype()));

    tensorflow::Tensor tensor(data_type, std::move(tensor_shape));

    // IFRT's return does not contain sufficient information; so we use
    // sharding spec from metadata.
    VLOG(2) << "Output sharding: " << array_for_copy->sharding().DebugString();
    VLOG(2) << "Metadata sharding: " << metadata_retval.sharding();
    TF_ASSIGN_OR_RETURN(auto hlo_sharding, xla::HloSharding::FromProto(
                                               metadata_retval.sharding()));

    // TODO(b/313922535): support sharded outputs.
    if (!hlo_sharding.IsReplicated()) {
      return absl::UnimplementedError(
          absl::StrCat("Only supported replicated output. But got ",
                       hlo_sharding.ToString()));
    }

    TF_ASSIGN_OR_RETURN(auto fully_replicated_array,
                        array_for_copy->FullyReplicatedShard(
                            xla::ifrt::ArrayCopySemantics::kDonateInput));

    if (fully_replicated_array->shape() !=
        xla::ifrt::Shape(tensor.shape().dim_sizes())) {
      return absl::UnimplementedError(
          absl::StrCat("Not fully replicated output. Expected ",
                       tensor.shape().DebugString(), " but got ",
                       fully_replicated_array->shape().DebugString()));
    }

    xla::ifrt::Future<absl::Status> copy_future =
        fully_replicated_array->CopyToHostBuffer(
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
