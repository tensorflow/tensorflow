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

#include "tensorflow/compiler/xla/python/py_host_callback.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/pjrt/host_callback.h"
#include "tensorflow/compiler/xla/python/callback.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/python/ifrt/host_callback.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

char PyCpuLoadedHostCallback::ID = 0;
char PyHostSendAndRecvLoadedHostCallback::ID = 0;

namespace {

StatusOr<std::vector<CpuCallback::Arg>> CreateCallbackArgs(
    absl::Span<const Shape> operand_shapes) {
  std::vector<CpuCallback::Arg> callback_args(operand_shapes.size());
  for (int i = 0; i < operand_shapes.size(); ++i) {
    Shape shape = operand_shapes[i];

    if (shape.IsArray()) {
      Shape layout =
          (shape.has_layout() ? shape
                              : LayoutUtil::GetWithDefaultLayout(shape));
      callback_args[i].dims.resize(shape.dimensions_size());
      absl::c_copy(shape.dimensions(), callback_args[i].dims.begin());
      callback_args[i].strides = ByteStridesForShape(layout);
      callback_args[i].type = shape.element_type();
      callback_args[i].size_in_bytes = ShapeUtil::ByteSizeOf(layout);
      TF_ASSIGN_OR_RETURN(callback_args[i].dtype,
                          PrimitiveTypeToDtype(shape.element_type()));
    } else if (shape.IsToken()) {
      callback_args[i].type = TOKEN;
    } else {
      return InvalidArgument(
          "Only array and token arguments to Python callbacks are supported, "
          "got %s",
          shape.ToString());
    }
  }
  return callback_args;
}

StatusOr<std::vector<CpuCallback::Result>> CreateCallbackResults(
    absl::Span<const Shape> result_shapes) {
  std::vector<CpuCallback::Result> callback_results(result_shapes.size());
  for (int i = 0; i < result_shapes.size(); ++i) {
    if (result_shapes[i].IsArray()) {
      const Shape& shape =
          result_shapes[i].has_layout()
              ? result_shapes[i]
              : LayoutUtil::GetWithDefaultLayout(result_shapes[i]);
      callback_results[i].expected_dims.resize(shape.dimensions_size());
      absl::c_copy(shape.dimensions(),
                   callback_results[i].expected_dims.begin());
      callback_results[i].expected_strides = ByteStridesForShapeInt64(shape);
      callback_results[i].type = shape.element_type();
      callback_results[i].size_in_bytes = ShapeUtil::ByteSizeOf(shape);
      callback_results[i].reversed_layout.resize(shape.dimensions_size());
      absl::c_reverse_copy(shape.layout().minor_to_major(),
                           callback_results[i].reversed_layout.begin());
    } else if (result_shapes[i].IsToken()) {
      callback_results[i].type = TOKEN;
    } else {
      return InvalidArgument(
          "Only array and token return values from Python callbacks are "
          "supported, got %s",
          result_shapes[i].ToString());
    }
  }
  return callback_results;
}

}  // namespace

StatusOr<tsl::RCReference<PyCpuLoadedHostCallback>>
PyCpuLoadedHostCallback::Create(ifrt::Client* ifrt_client,
                                pybind11::function callable,
                                absl::Span<const Shape> operand_shapes,
                                absl::Span<const Shape> result_shapes) {
  ifrt::PlatformId platform_id = ifrt_client->platform_id();
  if (platform_id != GpuId() && platform_id != CpuId()) {
    return Unimplemented("CpuCallback supports CPU and GPU only");
  }

  TF_ASSIGN_OR_RETURN(auto callback_args, CreateCallbackArgs(operand_shapes));
  TF_ASSIGN_OR_RETURN(auto callback_results,
                      CreateCallbackResults(result_shapes));

  // `callable` will be destroyed safely with `PythonRefManager` when
  // `CpuCallback` is destroyed.
  auto cpu_callback = std::make_unique<CpuCallback>(
      std::move(callable), callback_args, callback_results);
  return tsl::RCReference<PyCpuLoadedHostCallback>(
      tsl::MakeRef<PyCpuLoadedHostCallback>(ifrt_client,
                                            std::move(cpu_callback)));
}

StatusOr<std::string> PyCpuLoadedHostCallback::Serialize() const {
  return Unimplemented(
      "PyHostSendAndRecvLoadedHostCallback serialization is not supported");
}

StatusOr<tsl::RCReference<PyHostSendAndRecvLoadedHostCallback>>
PyHostSendAndRecvLoadedHostCallback::Create(
    ifrt::Client* ifrt_client, pybind11::function callable,
    absl::Span<const Shape> operand_shapes,
    absl::Span<const Shape> result_shapes,
    absl::Span<const uint16_t> send_channel_ids,
    absl::Span<const uint16_t> recv_channel_ids) {
  TF_ASSIGN_OR_RETURN(auto callback_args, CreateCallbackArgs(operand_shapes));
  TF_ASSIGN_OR_RETURN(auto callback_results,
                      CreateCallbackResults(result_shapes));

  // `callable` will be destroyed safely with `PythonRefManager` when
  // `CpuCallback` is destroyed.
  auto cpu_callback =
      std::make_shared<CpuCallback>(callable, callback_args, callback_results);

  auto host_callback = std::make_unique<HostCallback>();

  auto assign_arg_info = [](absl::Span<const xla::Shape> shapes,
                            absl::Span<const uint16_t> channel_ids,
                            std::vector<HostCallbackArgInfo>& arg_infos) {
    DCHECK_EQ(shapes.size(), channel_ids.size());
    arg_infos.reserve(shapes.size());
    for (int i = 0; i < shapes.size(); ++i) {
      HostCallbackArgInfo host_callback_arg_info;
      host_callback_arg_info.channel_id = channel_ids[i];
      const auto& shape = shapes[i];
      Shape layout =
          (shape.has_layout() ? shape
                              : LayoutUtil::GetWithDefaultLayout(shape));
      host_callback_arg_info.shape = layout;
      arg_infos.push_back(std::move(host_callback_arg_info));
    }
  };

  assign_arg_info(operand_shapes, send_channel_ids, host_callback->operands);
  assign_arg_info(result_shapes, recv_channel_ids, host_callback->results);

  host_callback->callback = [cpu_callback = std::move(cpu_callback)](
                                void** outputs, void** inputs) {
    return cpu_callback->PrepareAndCall(outputs, inputs);
  };
  return tsl::RCReference<PyHostSendAndRecvLoadedHostCallback>(
      tsl::MakeRef<PyHostSendAndRecvLoadedHostCallback>(
          ifrt_client, std::move(host_callback), callable, operand_shapes,
          result_shapes, send_channel_ids, recv_channel_ids));
}

PyHostSendAndRecvLoadedHostCallback::PyHostSendAndRecvLoadedHostCallback(
    ifrt::Client* ifrt_client,
    std::unique_ptr<xla::HostCallback> xla_host_callback,
    pybind11::function callable, absl::Span<const Shape> operand_shapes,
    absl::Span<const Shape> result_shapes,
    absl::Span<const uint16_t> send_channel_ids,
    absl::Span<const uint16_t> recv_channel_ids)
    : llvm::RTTIExtends<PyHostSendAndRecvLoadedHostCallback,
                        ifrt::PjRtHostSendAndRecvLoadedHostCallback>(
          ifrt_client, std::move(xla_host_callback)),
      callable_(std::move(callable)),
      operand_shapes_(operand_shapes.begin(), operand_shapes.end()),
      result_shapes_(result_shapes.begin(), result_shapes.end()),
      send_channel_ids_(send_channel_ids.begin(), send_channel_ids.end()),
      recv_channel_ids_(recv_channel_ids.begin(), recv_channel_ids.end()) {}

PyHostSendAndRecvLoadedHostCallback::~PyHostSendAndRecvLoadedHostCallback() {
  GlobalPyRefManager()->AddGarbage(
      absl::MakeSpan(static_cast<pybind11::object*>(&callable_), 1));
}

StatusOr<std::string> PyHostSendAndRecvLoadedHostCallback::Serialize() const {
  return Unimplemented(
      "PyHostSendAndRecvLoadedHostCallback serialization is not yet "
      "implemented");
}

}  // namespace xla
