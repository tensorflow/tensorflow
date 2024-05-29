/* Copyright 2022 The OpenXLA Authors.

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
#include "xla/python/custom_call_sharding.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_custom_partitioner_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/custom_partition_callback.h"
#include "xla/python/inspect_sharding.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace nb = ::nanobind;

class PyCustomCallPartitionerCallbacks {
 public:
  PyCustomCallPartitionerCallbacks(nb::object prop_user_sharding,
                                   nb::object partition,
                                   nb::object infer_sharding_from_operands)
      : prop_user_sharding_(prop_user_sharding),
        partition_(partition),
        infer_sharding_from_operands_(infer_sharding_from_operands) {
    callbacks_.version = 0;
    callbacks_.private_data = this;
    callbacks_.dtor = +[](JAX_CustomCallPartitioner_Callbacks* self) {
      delete GetSelfPtr(self);
    };
    callbacks_.partition = +[](JAX_CustomCallPartitioner_Callbacks* self,
                               JAX_CustomCallPartitioner_Partition_Args* args) {
      jax::PopulateResults(GetSelfPtr(self)->CallPartition(args), args);
    };
    callbacks_.infer_sharding =
        +[](JAX_CustomCallPartitioner_Callbacks* self,
            JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args) {
          jax::PopulateResults(
              GetSelfPtr(self)->CallInferShardingFromOperands(args), args);
        };
    callbacks_.propagate_user_sharding =
        +[](JAX_CustomCallPartitioner_Callbacks* self,
            JAX_CustomCallPartitioner_PropagateUserSharding_Args* args) {
          jax::PopulateResults(
              GetSelfPtr(self)->CallPropagateUserSharding(args), args);
        };
  }

  absl::StatusOr<
      std::tuple<std::string, std::vector<xla::HloSharding>, xla::HloSharding>>
  CallPartition(JAX_CustomCallPartitioner_Partition_Args* args) const {
    if (args->header.api_version != 0) {
      return absl::InternalError("API version mismatch.");
    }
    TF_ASSIGN_OR_RETURN(auto args_tuple, jax::ReadArgs(args));
    std::vector<xla::Shape> shapes = std::move(std::get<0>(args_tuple));
    std::vector<std::optional<xla::HloSharding>> shardings =
        std::move(std::get<1>(args_tuple));
    xla::Shape result_shape = std::move(std::get<2>(args_tuple));
    std::optional<xla::HloSharding> result_sharding =
        std::move(std::get<3>(args_tuple));
    std::string_view backend_config = std::move(std::get<4>(args_tuple));

    {
      nb::gil_scoped_acquire gil;
      try {
        auto py_result =
            partition_(shapes, shardings, result_shape, result_sharding,
                       nb::bytes(backend_config.data(), backend_config.size()));
        try {
          auto [ir, arg_shardings, result_sharding] = nb::cast<
              std::tuple<nb::bytes, std::vector<HloSharding>, HloSharding>>(
              py_result);
          if (arg_shardings.size() != args->num_args) {
            return xla::Internal(
                "Shardings returned from partitioning: lengths must match: %d "
                "vs %d",
                arg_shardings.size(), args->num_args);
          }
          return std::make_tuple(std::string(ir.c_str(), ir.size()),
                                 std::move(arg_shardings),
                                 std::move(result_sharding));
        } catch (const nb::cast_error& e) {
          return xla::Internal(
              "Shardings returned from partitioning: expected "
              "Tuple[bytes, List[HloSharding], HloSharding] got: %s",
              nb::cast<std::string_view>(nb::repr(py_result)));
        }
      } catch (const nb::python_error& e) {
        return xla::Internal("custom_partitioner: %s", e.what());
      }
    }
  }

  absl::StatusOr<std::optional<xla::HloSharding>> CallInferShardingFromOperands(
      JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args) const {
    if (args->header.api_version != 0) {
      return absl::InternalError("API version mismatch.");
    }
    TF_ASSIGN_OR_RETURN(auto args_tuple, jax::ReadArgs(args));
    std::vector<xla::Shape> arg_shapes = std::move(std::get<0>(args_tuple));
    std::vector<std::optional<xla::HloSharding>> arg_shardings =
        std::move(std::get<1>(args_tuple));
    xla::Shape result_shape = std::move(std::get<2>(args_tuple));
    std::string_view backend_config = std::move(std::get<3>(args_tuple));

    std::optional<HloSharding> result;
    nb::gil_scoped_acquire gil;
    try {
      auto py_result = infer_sharding_from_operands_(
          arg_shapes, arg_shardings, result_shape,
          nb::bytes(backend_config.data(), backend_config.size()));
      if (py_result.is_none()) {
        return std::nullopt;
      }
      return nb::cast<HloSharding>(py_result);
    } catch (const nb::python_error& e) {
      return xla::Internal("custom_partitioner: %s", e.what());
    }
  }

  absl::StatusOr<xla::HloSharding> CallPropagateUserSharding(
      JAX_CustomCallPartitioner_PropagateUserSharding_Args* args) const {
    if (args->header.api_version != 0) {
      return absl::InternalError("API version mismatch.");
    }
    TF_ASSIGN_OR_RETURN(auto args_tuple, jax::ReadArgs(args));
    xla::HloSharding result_sharding = std::move(std::get<0>(args_tuple));
    xla::Shape result_shape = std::move(std::get<1>(args_tuple));
    std::string_view backend_config = std::move(std::get<2>(args_tuple));

    nb::gil_scoped_acquire gil;
    try {
      // TODO(parkers): expand this API to handle the `user` sharding.
      // The user is used when the custom call returns a Tuple and
      // the user is a get-tuple-element. In this case we must update only
      // part of the sharding spec.
      auto result = nb::cast<HloSharding>(prop_user_sharding_(
          result_sharding, result_shape,
          nb::bytes(backend_config.data(), backend_config.size())));
      return result;
    } catch (const nb::python_error& e) {
      return xla::Internal("custom_partitioner: %s", e.what());
    }
  }

  JAX_CustomCallPartitioner_Callbacks* callbacks() { return &callbacks_; }

 private:
  static PyCustomCallPartitionerCallbacks* GetSelfPtr(
      JAX_CustomCallPartitioner_Callbacks* callbacks) {
    return reinterpret_cast<PyCustomCallPartitionerCallbacks*>(
        callbacks->private_data);
  }

  JAX_CustomCallPartitioner_Callbacks callbacks_;
  nb::object prop_user_sharding_;
  nb::object partition_;
  nb::object infer_sharding_from_operands_;
};

namespace {

void CallInspectSharding(void* obj, JAX_InspectSharding_Callback_Args* args) {
  std::optional<xla::HloSharding> arg = jax::InspectShardingReadArgs(args);
  if (!arg.has_value()) {
    return;
  }
  try {
    nb::gil_scoped_acquire gil;
    nb::handle(reinterpret_cast<PyObject*>(obj))(*std::move(arg));
  } catch (const nb::python_error& e) {
    jax::InspectShardingSetError(args, std::string(e.what()));
  }
}

}  // namespace

void BuildCustomCallShardingPybindAPI(nb::module_& m) {
  m.def(
      "register_custom_call_partitioner",
      [](std::string name, nb::object prop_user_sharding, nb::object partition,
         nb::object infer_sharding_from_operands,
         bool can_side_effecting_have_replicated_sharding,
         std::optional<nb::capsule> c_api) {
        auto* c_fns =
            (new PyCustomCallPartitionerCallbacks(prop_user_sharding, partition,
                                                  infer_sharding_from_operands))
                ->callbacks();
        c_fns->can_side_effecting_have_replicated_sharding =
            can_side_effecting_have_replicated_sharding;
        if (!c_api.has_value()) {
          RegisterCustomCallPartitioner(
              name, jax::CreateCApiCustomCallPartitioner(c_fns));
          return;
        }

        if (std::string_view(c_api->name()) != "pjrt_c_api") {
          throw absl::InvalidArgumentError(
              "Argument to register_custom_call_partitioner was not a "
              "pjrt_c_api capsule.");
        }
        auto* c_api_value = static_cast<const PJRT_Api*>(c_api->data());
        PJRT_Custom_Partitioner_Extension* extension =
            pjrt::FindExtension<PJRT_Custom_Partitioner_Extension>(
                c_api_value,
                PJRT_Extension_Type::PJRT_Extension_Type_Custom_Partitioner);
        if (extension == nullptr) {
          return;
        }
        PJRT_Register_Custom_Partitioner_Args args;
        args.struct_size = PJRT_Register_Custom_Partitioner_Args_STRUCT_SIZE;
        args.name = name.c_str();
        args.name_size = name.size();
        args.callbacks = c_fns;
        PJRT_Error* error =
            reinterpret_cast<const PJRT_Custom_Partitioner_Extension*>(
                extension)
                ->register_custom_partitioner(&args);
        std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error_ptr(
            error, pjrt::MakeErrorDeleter(c_api_value));
        ThrowIfError(pjrt::PjrtErrorToStatus(error_ptr.get(), c_api_value));
      },
      R"(Registers a partitioner for a custom-call operation.

Args:
  name: custom_call_target to match.
  prop_user_sharding: Custom backwards sharding propagation rule.
     Takes result sharding and returns the instruction sharding.
  partition: Lowering rule. Takes operand and result shardings and returns
     a generated HLO and sharding specs. The spmd lowerer first reshards
     to match the returned sharding specs and then inserts the generated hlo.
  infer_sharding_from_operands: Custom forwards sharding propagation rule.
     Takes operand sharding and returns the instruction sharding.
  can_side_effecting_have_replicated_sharding: Side effecting ops are not
     allowed to have replicated sharding. Pass true to disable this check.
  c_api: Optional `PJRT_Api*` if it is called with a plugin. This is safe to
     call on plugins that do not implement the custom partitioner extension
)",
      nb::arg("name"), nb::arg("prop_user_sharding"), nb::arg("partition"),
      nb::arg("infer_sharding_from_operands"),
      nb::arg("can_side_effecting_have_replicated_sharding") = false,
      nb::arg("c_api").none() = std::nullopt);
  m.def("encode_inspect_sharding_callback",
        [](nb::object handler) -> nb::bytes {
          JAX_InspectSharding_Callback cb;
          cb.call = &CallInspectSharding;
          cb.data = handler.ptr();
          char bytes[sizeof(JAX_InspectSharding_Callback)];
          std::memcpy(&bytes, &cb, sizeof(JAX_InspectSharding_Callback));
          return nb::bytes(bytes, sizeof(JAX_InspectSharding_Callback));
        });

  nb::module_ hlo_sharding_util_m = m.def_submodule(
      "hlo_sharding_util", "Utilities for manipulating HloSharding.");
  hlo_sharding_util_m.def(
      "PartiallyReplicateTiledShardingOnDims",
      [](const HloSharding& sharding, std::vector<int64_t> dims) {
        return hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            sharding, dims);
      });
}

}  // namespace xla
