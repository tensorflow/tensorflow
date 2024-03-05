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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/custom_partition_callback.h"
#include "xla/python/inspect_sharding.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

namespace py = ::pybind11;

class PyCustomCallPartitionerCallbacks {
 public:
  PyCustomCallPartitionerCallbacks(py::object prop_user_sharding,
                                   py::object partition,
                                   py::object infer_sharding_from_operands)
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
      py::gil_scoped_acquire gil;
      try {
        auto py_result = partition_(shapes, shardings, result_shape,
                                    result_sharding, py::bytes(backend_config));
        try {
          auto result = py::cast<
              std::tuple<std::string, std::vector<HloSharding>, HloSharding>>(
              py_result);
          if (std::get<1>(result).size() != args->num_args) {
            return xla::Internal(
                "Shardings returned from partitioning: lengths must match: %d "
                "vs %d",
                std::get<1>(result).size(), args->num_args);
          }
          return result;
        } catch (const py::cast_error& e) {
          return xla::Internal(
              "Shardings returned from partitioning: expected "
              "Tuple[bytes, List[HloSharding], HloSharding] got: %s",
              py::repr(py_result));
        }
      } catch (const pybind11::error_already_set& e) {
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
    py::gil_scoped_acquire gil;
    try {
      auto py_result = infer_sharding_from_operands_(
          arg_shapes, arg_shardings, result_shape, py::bytes(backend_config));
      if (py_result.is_none()) {
        return std::nullopt;
      }
      return py::cast<HloSharding>(py_result);
    } catch (const pybind11::error_already_set& e) {
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

    py::gil_scoped_acquire gil;
    try {
      // TODO(parkers): expand this API to handle the `user` sharding.
      // The user is used when the custom call returns a Tuple and
      // the user is a get-tuple-element. In this case we must update only
      // part of the sharding spec.
      auto result = py::cast<HloSharding>(prop_user_sharding_(
          result_sharding, result_shape, py::bytes(backend_config)));
      return result;
    } catch (const pybind11::error_already_set& e) {
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
  py::object prop_user_sharding_;
  py::object partition_;
  py::object infer_sharding_from_operands_;
};

namespace {

void CallInspectSharding(void* obj, JAX_InspectSharding_Callback_Args* args) {
  std::optional<xla::HloSharding> arg = jax::InspectShardingReadArgs(args);
  if (!arg.has_value()) {
    return;
  }
  try {
    py::gil_scoped_acquire gil;
    py::handle(reinterpret_cast<PyObject*>(obj))(*std::move(arg));
  } catch (const pybind11::error_already_set& e) {
    jax::InspectShardingSetError(args, std::string(e.what()));
  }
}

}  // namespace

void BuildCustomCallShardingPybindAPI(pybind11::module& m) {
  m.def(
      "register_custom_call_partitioner",
      [](std::string name, py::object prop_user_sharding, py::object partition,
         py::object infer_sharding_from_operands,
         bool can_side_effecting_have_replicated_sharding) {
        auto* c_fns =
            (new PyCustomCallPartitionerCallbacks(prop_user_sharding, partition,
                                                  infer_sharding_from_operands))
                ->callbacks();
        c_fns->can_side_effecting_have_replicated_sharding =
            can_side_effecting_have_replicated_sharding;
        RegisterCustomCallPartitioner(
            name, jax::CreateCApiCustomCallPartitioner(c_fns));
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
)",
      py::arg("name"), py::arg("prop_user_sharding"), py::arg("partition"),
      py::arg("infer_sharding_from_operands"),
      py::arg("can_side_effecting_have_replicated_sharding") = false);
  m.def("encode_inspect_sharding_callback",
        [](py::object handler) -> py::bytes {
          JAX_InspectSharding_Callback cb;
          cb.call = &CallInspectSharding;
          cb.data = handler.ptr();
          char bytes[sizeof(JAX_InspectSharding_Callback)];
          memcpy(&bytes, &cb, sizeof(JAX_InspectSharding_Callback));
          return py::bytes(bytes, sizeof(JAX_InspectSharding_Callback));
        });

  py::module hlo_sharding_util_m = m.def_submodule(
      "hlo_sharding_util", "Utilities for manipulating HloSharding.");
  hlo_sharding_util_m.def(
      "PartiallyReplicateTiledShardingOnDims",
      [](const HloSharding& sharding, std::vector<int64_t> dims) {
        return hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            sharding, dims);
      });
}

}  // namespace xla
