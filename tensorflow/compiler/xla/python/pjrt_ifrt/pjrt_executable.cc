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

#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_executable.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_array.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

// Returns the op sharding of the root instruction in the entry computation.
StatusOr<const xla::HloInstructionProto*> FindRootInstruction(
    const HloModuleProto& proto) {
  for (const auto& computation : proto.computations()) {
    if (computation.id() == proto.entry_computation_id()) {
      for (const auto& instruction : computation.instructions()) {
        if (instruction.id() == computation.root_id()) {
          return &instruction;
        }
      }
    }
  }
  return InvalidArgument("Entry computation not found");
}

}  // namespace

char PjRtExecutable::ID = 0;
char PjRtLoadedExecutable::ID = 0;

StatusOr<std::unique_ptr<Executable>> PjRtExecutable::Create(
    std::unique_ptr<xla::PjRtExecutable> pjrt_executable) {
  return std::unique_ptr<Executable>(new PjRtExecutable(
      std::shared_ptr<xla::PjRtExecutable>(pjrt_executable.release())));
}

StatusOr<std::unique_ptr<Executable>> PjRtExecutable::Create(
    std::shared_ptr<xla::PjRtExecutable> pjrt_executable) {
  return std::unique_ptr<Executable>(
      new PjRtExecutable(std::move(pjrt_executable)));
}

StatusOr<std::optional<std::string>> PjRtExecutable::Fingerprint() const {
  DCHECK(this);
  return pjrt_executable_->FingerprintExecutable();
}

StatusOr<std::string> PjRtExecutable::Serialize() const {
  DCHECK(this);
  return pjrt_executable_->SerializeExecutable();
}

StatusOr<std::unique_ptr<LoadedExecutable>> PjRtLoadedExecutable::Create(
    PjRtClient* client,
    std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable) {
  return Create(client, std::shared_ptr<xla::PjRtLoadedExecutable>(
                            pjrt_loaded_executable.release()));
}

StatusOr<std::unique_ptr<LoadedExecutable>> PjRtLoadedExecutable::Create(
    PjRtClient* client,
    std::shared_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable) {
  // TODO(hyeontaek): We should request output sharding instead of the entire
  // HLO modules once PjRt supports it.
  // TODO(hyeontaek): We would not need to use GetHloModules() if
  // PjRtLoadedExecutable can return the result shape and op sharding when
  // output sharding propagation is not used.
  VLOG(3) << "PjRtLoadedExecutable::Create";
  VLOG(3) << "Requesting GetHloModules";
  TF_ASSIGN_OR_RETURN(auto hlo_modules,
                      pjrt_loaded_executable->GetHloModules());
  if (hlo_modules.empty()) {
    return FailedPrecondition("No HLO module found");
  }
  const auto& hlo_module = hlo_modules.front();
  // result_shape already contains per-device shapes. Do not use HLO sharding
  // (e.g., from hlo_module->spmd_output_sharding()), which would accidentally
  // apply sharding twice.
  const xla::Shape& result_shape = hlo_module->result_shape();
  return CreateInternal(client, std::move(pjrt_loaded_executable), result_shape,
                        /*result_hlo_sharding=*/nullptr);
}

StatusOr<std::unique_ptr<LoadedExecutable>> PjRtLoadedExecutable::Create(
    PjRtClient* client, const XlaComputation& computation,
    CompileOptions options) {
  VLOG(3) << "PjRtLoadedExecutable::Create";
  VLOG(3) << computation.proto().DebugString();
  VLOG(3) << options.ToProto()->DebugString();
  const auto& build_options = options.executable_build_options;
  const bool auto_spmd_partitioning =
      build_options.use_spmd_partitioning() &&
      build_options.num_partitions() > 1 &&
      (build_options.use_auto_spmd_partitioning() ||
       build_options.allow_spmd_sharding_propagation_to_output());
  TF_ASSIGN_OR_RETURN(
      auto pjrt_loaded_executable,
      client->pjrt_client()->Compile(computation, std::move(options)));

  if (auto_spmd_partitioning) {
    // TODO(hyeontaek): We should request output shapes and shardings instead of
    // the entire HLO modules once PjRt supports it.
    VLOG(3) << "Requesting GetHloModules";
    TF_ASSIGN_OR_RETURN(auto hlo_modules,
                        pjrt_loaded_executable->GetHloModules());
    if (hlo_modules.empty()) {
      return FailedPrecondition("No HLO module found");
    }
    const auto& hlo_module = hlo_modules.front();
    // result_shape already contains per-device shapes. Do not use HLO sharding
    // (e.g., from hlo_module->spmd_output_sharding()), which would accidentally
    // apply sharding twice.
    const xla::Shape& result_shape = hlo_module->result_shape();
    return CreateInternal(client, std::move(pjrt_loaded_executable),
                          result_shape,
                          /*result_hlo_sharding=*/nullptr);
  } else {
    VLOG(3) << "Not requesting GetHloModules";
    TF_ASSIGN_OR_RETURN(const auto* root_instruction,
                        FindRootInstruction(computation.proto()));
    const xla::Shape result_shape(root_instruction->shape());
    const xla::HloSharding* result_hlo_sharding = nullptr;
    std::optional<xla::HloSharding> result_hlo_sharding_holder;
    if (root_instruction->has_sharding()) {
      TF_ASSIGN_OR_RETURN(
          result_hlo_sharding_holder,
          xla::HloSharding::FromProto(root_instruction->sharding()));
      result_hlo_sharding = &*result_hlo_sharding_holder;
    }
    return CreateInternal(client, std::move(pjrt_loaded_executable),
                          result_shape, result_hlo_sharding);
  }
}

StatusOr<std::unique_ptr<LoadedExecutable>>
PjRtLoadedExecutable::CreateInternal(
    PjRtClient* client,
    std::shared_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable,
    const xla::Shape& result_shape,
    const xla::HloSharding* result_hlo_sharding) {
  DeviceList devices(
      DeviceList::Devices(pjrt_loaded_executable->addressable_devices().begin(),
                          pjrt_loaded_executable->addressable_devices().end()));
  if (devices.empty()) {
    return InvalidArgument("At least one device is required");
  }
  std::vector<DType> output_dtypes;
  std::vector<Shape> output_shapes;
  std::vector<std::shared_ptr<const Sharding>> output_shardings;

  auto append_arg = [&](const xla::Shape& shape,
                        const xla::HloSharding* sharding) -> Status {
    TF_ASSIGN_OR_RETURN(auto dtype, ToDType(shape.element_type()));
    output_dtypes.push_back(dtype);
    output_shapes.push_back(Shape(shape.dimensions()));

    CHECK(shape.IsArray());

    xla::Shape tile_shape;
    if (sharding != nullptr) {
      CHECK(!sharding->IsTuple());
      tile_shape = sharding->TileShape(shape);
    } else {
      tile_shape = shape;
    }
    std::vector<Shape> per_device_shapes(
        /*n=*/pjrt_loaded_executable->addressable_devices().size(),
        /*v=*/Shape(tile_shape.dimensions()));
    output_shardings.push_back(OpaqueSharding::Create(
        devices, OpaqueSharding::MakeDisassembleFuncFromShapes(
                     std::move(per_device_shapes))));
    return OkStatus();
  };
  auto append_token = [&] {
    output_dtypes.push_back(DType(DType::kToken));
    output_shapes.push_back(Shape({}));
    output_shardings.push_back(OpaqueSharding::Create(devices));
  };

  if (result_shape.IsArray()) {
    output_dtypes.reserve(1);
    output_shapes.reserve(1);
    output_shardings.reserve(1);
    TF_RETURN_IF_ERROR(append_arg(result_shape, result_hlo_sharding));
  } else if (result_shape.IsToken()) {
    output_dtypes.reserve(1);
    output_shapes.reserve(1);
    output_shardings.reserve(1);
    append_token();
  } else if (result_shape.IsTuple()) {
    output_dtypes.reserve(result_shape.tuple_shapes().size());
    output_shapes.reserve(result_shape.tuple_shapes().size());
    output_shardings.reserve(result_shape.tuple_shapes().size());
    if (result_hlo_sharding != nullptr &&
        (!result_hlo_sharding->IsTuple() ||
         result_hlo_sharding->tuple_elements().size() !=
             result_shape.tuple_shapes().size())) {
      return FailedPrecondition(
          "Output sharding is inconsistent with the tuple result");
    }
    for (int i = 0; i < result_shape.tuple_shapes().size(); ++i) {
      const auto& element_shape = result_shape.tuple_shapes(i);
      if (element_shape.IsArray()) {
        const xla::HloSharding* element_hlo_sharding = nullptr;
        if (result_hlo_sharding != nullptr) {
          element_hlo_sharding = &result_hlo_sharding->tuple_elements()[i];
          if (element_hlo_sharding->IsTuple()) {
            return FailedPrecondition(
                "Output sharding is inconsistent with the tuple result");
          }
        }
        TF_RETURN_IF_ERROR(append_arg(element_shape, element_hlo_sharding));
      } else if (element_shape.IsToken()) {
        append_token();
      } else {
        return FailedPrecondition(
            "The tuple element is not a supported type (array, token)");
      }
    }
  } else {
    return FailedPrecondition(
        "The computation result is not a support type (array, token, tuple)");
  }

  return std::unique_ptr<LoadedExecutable>(new PjRtLoadedExecutable(
      client, std::move(pjrt_loaded_executable), std::move(devices),
      std::move(output_dtypes), std::move(output_shapes),
      std::move(output_shardings)));
}

StatusOr<PjRtLoadedExecutable::ExecuteResult> PjRtLoadedExecutable::Execute(
    absl::Span<Array* const> args, const ExecuteOptions& options,
    std::optional<DeviceList> devices) {
  DCHECK(this);
  // TODO(hyeontaek): Check input sharding consistency.

  // Convert an Array vector into 2-level PjRtBuffer vectors, optionally copying
  // to new devices.
  std::vector<std::vector<PjRtBuffer*>> argument_handles;
  std::vector<std::unique_ptr<PjRtBuffer>> owned_buffers;

  const int num_computations = devices_.size();
  argument_handles.resize(num_computations);
  for (int i = 0; i < num_computations; ++i) {
    argument_handles[i].reserve(args.size());
  }
  for (int i = 0; i < args.size(); ++i) {
    auto* pjrt_array = llvm::dyn_cast_or_null<PjRtArray>(args[i]);
    if (!pjrt_array) {
      return InvalidArgument(
          "Only PjRtArray is supported, but argument %d is %s", i,
          pjrt_array->DebugString());
    }
    int j = 0;
    // TODO(hyeontaek): Check pjrt_array->pjrt_buffers().size() ==
    // num_computations
    for (const auto& pjrt_buffer : pjrt_array->pjrt_buffers()) {
      argument_handles[j].push_back(pjrt_buffer.get());
      ++j;
    }
  }

  const bool portable_execution = devices.has_value();
  Device* portable_execution_device = devices_.front();
  if (portable_execution) {
    if (devices->size() != 1) {
      return InvalidArgument(
          "Only single-shard portable execution is supported");
    }
    portable_execution_device = devices->front();
  }

  if (portable_execution) {
    if (!argument_handles[0].empty()) {
      portable_execution_device = argument_handles[0][0]->device();
    } else {
      // Cannot infer the device from the input.
      // TODO(hyeontaek): Probably we should take devices as an argument?
      portable_execution_device = devices_.front();
    }
  }

  const bool returned_future_supported =
      pjrt_loaded_executable_->IsReturnedFutureSupported();

  // Execute the computation.
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> pjrt_outputs;
  ExecuteResult result;
  if (portable_execution) {
    std::optional<PjRtFuture<Status>> returned_pjrt_future;
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<PjRtBuffer>> single_device_pjrt_results,
        pjrt_loaded_executable_->ExecutePortable(
            argument_handles.front(), portable_execution_device, options,
            returned_pjrt_future, /*fill_future=*/returned_future_supported));

    pjrt_outputs.push_back(std::move(single_device_pjrt_results));
    if (returned_future_supported) {
      result.status = *std::move(returned_pjrt_future);
    } else {
      result.status = Future<Status>(OkStatus());
    }
  } else {
    std::optional<std::vector<PjRtFuture<Status>>> returned_pjrt_futures;
    if (returned_future_supported) {
      returned_pjrt_futures.emplace();
    }

    TF_ASSIGN_OR_RETURN(pjrt_outputs,
                        pjrt_loaded_executable_->Execute(
                            argument_handles, options, returned_pjrt_futures));

    if (returned_future_supported) {
      result.status = JoinFutures(absl::MakeSpan(*returned_pjrt_futures));
    } else {
      result.status = Future<Status>(OkStatus());
    }
  }

  // Convert 2-level PjRtBuffer vectors into an Array vector.
  std::vector<std::unique_ptr<Array>> outputs;
  // TODO(hyeontaek): Check output dtype/shape consistency with the actual
  // output.
  if (pjrt_outputs.size() != num_computations) {
    return FailedPrecondition(
        "Unexpected number of computations in outputs: %d vs. %d",
        pjrt_outputs.front().size(), num_computations);
  }
  const int num_outputs = pjrt_outputs.front().size();
  if (num_outputs != output_dtypes_.size()) {
    return FailedPrecondition("Unexpected number of outputs: %d vs. %d",
                              num_outputs, output_dtypes_.size());
  }
  outputs.reserve(num_outputs);
  std::shared_ptr<const Sharding> single_device_sharding;
  if (portable_execution) {
    single_device_sharding =
        SingleDeviceSharding::Create(portable_execution_device);
  }
  for (int i = 0; i < num_outputs; ++i) {
    PjRtArray::PjRtBuffers buffers;
    buffers.reserve(num_computations);
    for (int j = 0; j < num_computations; ++j) {
      buffers.push_back(
          std::shared_ptr<PjRtBuffer>(pjrt_outputs[j][i].release()));
    }
    std::shared_ptr<const Sharding> sharding;
    if (portable_execution) {
      sharding = single_device_sharding;
    } else {
      sharding = output_shardings_[i];
    }
    outputs.push_back(*PjRtArray::Create(client_, output_dtypes_[i],
                                         output_shapes_[i], std::move(sharding),
                                         std::move(buffers)));
  }
  result.outputs = std::move(outputs);
  return result;
}

StatusOr<std::optional<std::string>> PjRtLoadedExecutable::Fingerprint() const {
  DCHECK(this);
  return client_->pjrt_client()->ExecutableFingerprint(
      *pjrt_loaded_executable_);
}

StatusOr<std::string> PjRtLoadedExecutable::Serialize() const {
  DCHECK(this);
  return client_->pjrt_client()->SerializeExecutable(*pjrt_loaded_executable_);
}

Future<Status> PjRtLoadedExecutable::Delete() {
  DCHECK(this);
  pjrt_loaded_executable_->Delete();
  // TODO(hyeontaek): Return a correct future.
  return Future<Status>(OkStatus());
}

}  // namespace ifrt
}  // namespace xla
