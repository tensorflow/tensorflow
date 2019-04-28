/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xrt/client/xrt_client.h"

#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xrt/client/xrt_tf_client.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"

namespace tensorflow {

namespace {

// Deserializes a TensorProto containing a scalar string value.
xla::StatusOr<std::string> DeserializeTensorProtoAsString(
    const TensorProto& proto) {
  if (proto.dtype() != DT_STRING) {
    return errors::InvalidArgument("Tensors must be of type DT_STRING, got ",
                                   DataType_Name(proto.dtype()));
  }
  if (proto.tensor_shape().dim_size() != 0 ||
      proto.tensor_shape().unknown_rank()) {
    return errors::InvalidArgument("String tensor must be a scalar, got ",
                                   proto.tensor_shape().DebugString());
  }
  if (proto.string_val_size() > 0) {
    if (proto.string_val_size() != 1) {
      return errors::InvalidArgument(
          "Expected at most one string_val in TensorProto, got ",
          proto.string_val_size());
    }
    return proto.string_val(0);
  } else {
    std::string data;
    port::DecodeStringList(proto.tensor_content(), &data, 1);
    return data;
  }
}

// Deserializes a xla::Literal from a TensorProto.
xla::StatusOr<xla::Literal> DeserializeTensorProtoAsLiteral(
    const TensorProto& proto) {
  TF_ASSIGN_OR_RETURN(std::string data, DeserializeTensorProtoAsString(proto));
  xla::LiteralProto literal_proto;
  literal_proto.ParsePartialFromString(data);
  return xla::Literal::CreateFromProto(literal_proto);
}

}  // namespace

XrtBuffer::XrtBuffer(XrtTensorHandle handle, xla::Shape shape)
    : handle_(std::move(handle)), shape_(std::move(shape)) {}

XrtBuffer::~XrtBuffer() { Delete(); }

/*static*/ xla::StatusOr<std::shared_ptr<XrtBuffer>> XrtBuffer::FromLiteral(
    const std::shared_ptr<XrtContext>& context, int xrt_device_ordinal,
    const xla::LiteralSlice& literal) {
  xrt::XLAAllocation allocation;
  *allocation.mutable_value() = literal.ToProto();

  auto proto = absl::make_unique<TensorProto>();
  proto->set_dtype(DT_STRING);
  allocation.SerializeToString(proto->add_string_val());

  if (xrt_device_ordinal < 0 ||
      xrt_device_ordinal >= context->tf_device_ids().size()) {
    return errors::InvalidArgument("Invalid XRT device ordinal ",
                                   xrt_device_ordinal);
  }
  int tf_device_id = context->tf_device_ids().at(xrt_device_ordinal);
  XrtTensorHandle literal_handle =
      context->tf_context()->SendTensor(std::move(proto), tf_device_id,
                                        /*host_memory=*/true);

  XrtTensorHandle buffer_handle = std::move(context->tf_context()->EnqueueOp(
      "XRTAllocate", {&literal_handle}, /*output_arity=*/1, /*attrs=*/{},
      tf_device_id)[0]);

  return std::make_shared<XrtBuffer>(std::move(buffer_handle), literal.shape());
}

xla::StatusOr<xla::Literal> XrtBuffer::ToLiteral() const {
  TF_RET_CHECK(handle_.valid());
  XrtTensorHandle literal_handle = std::move(handle_.context()->EnqueueOp(
      "XRTReadLiteral", {&handle_}, /*output_arity=*/1, /*attrs=*/{},
      handle_.device_id())[0]);

  std::shared_ptr<XrtRecvTensorFuture> future =
      handle_.context()->RecvTensor(literal_handle, DT_STRING,
                                    /*host_memory=*/true);

  // Flush the queue to make sure the producers are dispatched before blocking
  // on the future.
  handle_.context()->FlushQueue();

  TF_ASSIGN_OR_RETURN(RecvTensorResponse * response, future->Get());
  VLOG(10) << "ToLiteral received tensor " << response->DebugString();
  TF_RET_CHECK(!response->is_dead());
  return DeserializeTensorProtoAsLiteral(response->tensor());
}

void XrtBuffer::Delete() {
  if (handle_.valid()) {
    handle_.context()->EnqueueOp("XRTReleaseAllocationHandle", {&handle_},
                                 /*output_arity=*/0,
                                 /*attrs=*/{}, handle_.device_id());
    handle_ = XrtTensorHandle();
  }
}

xla::StatusOr<std::vector<std::shared_ptr<XrtBuffer>>>
XrtBuffer::DestructureTuple() {
  TF_RET_CHECK(shape_.IsTuple());
  std::vector<std::shared_ptr<XrtBuffer>> output;
  output.reserve(shape_.tuple_shapes().size());
  for (int i = 0; i < shape_.tuple_shapes().size(); ++i) {
    TensorProto index_proto;
    index_proto.set_dtype(DT_INT32);
    index_proto.mutable_tensor_shape()->add_dim()->set_size(1);
    index_proto.add_int_val(i);
    XrtTensorHandle index =
        EnqueueConst(handle_.context().get(), handle_.device_id(), index_proto,
                     /*host_memory=*/true);
    XrtTensorHandle sub = std::move(
        handle_.context()->EnqueueOp("XRTSubTuple", {&handle_, &index},
                                     /*output_arity=*/1,
                                     /*attrs=*/{}, handle_.device_id())[0]);
    output.push_back(
        std::make_shared<XrtBuffer>(std::move(sub), shape_.tuple_shapes(i)));
  }
  return output;
}

/*static*/ xla::StatusOr<std::shared_ptr<XrtExecutable>> XrtExecutable::Compile(
    std::shared_ptr<XrtContext> context,
    const xla::HloModuleProto& hlo_module_proto,
    const std::vector<xla::Shape>& argument_shapes,
    const xla::Shape& result_shape, xla::DeviceAssignment device_assignment) {
  if (device_assignment.replica_count() <= 0 ||
      device_assignment.computation_count() <= 0) {
    return errors::InvalidArgument(
        "Device assignment must be non-empty; got ",
        device_assignment.replica_count(), " replicas and ",
        device_assignment.computation_count(), " computations per replica.");
  }

  // TODO(phawkins): add support for per-core argument and return shapes.
  TF_RET_CHECK(device_assignment.computation_count() == 1)
      << "Computation count != 1 not implemented";

  xrt::XLAComputation computation;
  computation.mutable_config()->set_num_replicas(
      device_assignment.replica_count());
  computation.mutable_config()->set_num_cores_per_replica(
      device_assignment.computation_count());

  xrt::DeviceAssignment* xrt_assignment =
      computation.mutable_config()->mutable_device_assignment();
  for (int computation = 0; computation < device_assignment.computation_count();
       ++computation) {
    xrt::DeviceAssignment::ComputationDevice* xrt_devices =
        xrt_assignment->add_computation_devices();
    for (int replica = 0; replica < device_assignment.replica_count();
         ++replica) {
      int xrt_device_ordinal = device_assignment(replica, computation);
      if (xrt_device_ordinal < 0 ||
          xrt_device_ordinal >= context->tf_device_ids().size()) {
        return errors::InvalidArgument("Invalid device ordinal in device ",
                                       "assignment: ", xrt_device_ordinal);
      }
      *xrt_devices->add_replica_devices() =
          context->device_mesh_coordinates().at(xrt_device_ordinal);
    }
  }

  xla::ProgramShape program_shape;
  for (const xla::Shape& shape : argument_shapes) {
    xla::Shape* param_shape = program_shape.add_parameters();
    *param_shape = shape;
    if (!xla::LayoutUtil::HasLayout(shape)) {
      xla::LayoutUtil::SetToDefaultLayout(param_shape);
    }
  }
  *program_shape.mutable_result() = result_shape;
  if (!xla::LayoutUtil::HasLayout(result_shape)) {
    xla::LayoutUtil::SetToDefaultLayout(program_shape.mutable_result());
  }
  *computation.mutable_config()->mutable_program_shape() =
      program_shape.ToProto();
  *computation.mutable_hlo_snapshot()->mutable_hlo()->mutable_hlo_module() =
      hlo_module_proto;

  auto proto = absl::make_unique<TensorProto>();
  proto->set_dtype(DT_STRING);
  computation.SerializeToString(proto->add_string_val());

  int xrt_device_ordinal_for_compilation = device_assignment(0, 0);
  int tf_device_id =
      context->tf_device_ids().at(xrt_device_ordinal_for_compilation);
  XrtTensorHandle computation_handle =
      context->tf_context()->SendTensor(std::move(proto), tf_device_id,
                                        /*host_memory=*/true);

  XrtTensorHandle executable_handle =
      std::move(context->tf_context()->EnqueueOp(
          "XRTCompile", {&computation_handle}, /*output_arity=*/2, /*attrs=*/{},
          tf_device_id)[0]);

  if (device_assignment.num_elements() > 1) {
    string wire_id = XrtGetUniqueWireID();
    int recv_tf_device_id = context->tf_context()->cpu_device_id();
    EnqueueSend(context->tf_context().get(), executable_handle, DT_INT64,
                recv_tf_device_id, wire_id, /*host_memory=*/true);
    executable_handle =
        EnqueueRecv(context->tf_context().get(), DT_INT64, tf_device_id,
                    recv_tf_device_id, wire_id, /*host_memory=*/true);
  }

  return std::make_shared<XrtExecutable>(
      std::move(context), std::move(executable_handle), program_shape,
      std::move(device_assignment));
}

XrtExecutable::XrtExecutable(std::shared_ptr<XrtContext> context,
                             XrtTensorHandle handle, xla::ProgramShape shape,
                             xla::DeviceAssignment device_assignment)
    : context_(std::move(context)),
      handle_(std::move(handle)),
      shape_(std::move(shape)),
      device_assignment_(std::move(device_assignment)) {}

XrtExecutable::~XrtExecutable() { Delete(); }

void XrtExecutable::Delete() {
  if (handle_.valid()) {
    handle_.context()->EnqueueOp("XRTReleaseCompilationHandle", {&handle_},
                                 /*output_arity=*/0,
                                 /*attrs=*/{}, handle_.device_id());
    handle_ = XrtTensorHandle();
  }
}

xla::StatusOr<std::shared_ptr<XrtBuffer>> XrtExecutable::Execute(
    const std::vector<std::shared_ptr<XrtBuffer>>& args) {
  TF_RET_CHECK(device_assignment_.replica_count() == 1 &&
               device_assignment_.computation_count() == 1)
      << device_assignment_.ToString();
  int xrt_device_ordinal = device_assignment_(0, 0);
  int tf_device_id = context_->tf_device_ids().at(xrt_device_ordinal);

  TensorProto config_proto;
  config_proto.set_dtype(DT_STRING);
  config_proto.add_string_val();
  XrtTensorHandle execution_config_handle =
      EnqueueConst(handle_.context().get(), tf_device_id, config_proto,
                   /*host_memory=*/true);

  protobuf::Map<string, AttrValue> attrs;
  attrs["Ninputs"] = MakeAttrValue(args.size());

  std::vector<const XrtTensorHandle*> inputs;
  inputs.reserve(args.size() + 2);
  inputs.push_back(&handle_);
  inputs.push_back(&execution_config_handle);
  for (const std::shared_ptr<XrtBuffer>& arg : args) {
    if (arg->handle().device_id() != tf_device_id) {
      return errors::InvalidArgument(
          "Input buffer to Execute() is not on the device for which the "
          "computation was compiled. Target device is ",
          tf_device_id, ", buffer is on device ", arg->handle().device_id());
    }
    inputs.push_back(&arg->handle());
  }

  XrtTensorHandle result_handle = std::move(handle_.context()->EnqueueOp(
      "XRTExecute", inputs, /*output_arity=*/1, attrs, tf_device_id)[0]);

  return std::make_shared<XrtBuffer>(std::move(result_handle), shape_.result());
}

xla::StatusOr<xla::Array2D<std::shared_ptr<XrtBuffer>>>
XrtExecutable::ExecuteReplicated(
    absl::Span<const xla::Array2D<std::shared_ptr<XrtBuffer>>> args) {
  if (args.size() != device_assignment_.computation_count()) {
    return errors::InvalidArgument(
        "Mismatched number of computation per replica between executable and "
        "arguments. Expected computations_per_replica=",
        device_assignment_.computation_count(),
        "; got computations_per_replica=", args.size());
  }

  for (int computation = 0;
       computation < device_assignment_.computation_count(); ++computation) {
    if (args[computation].n1() != device_assignment_.replica_count()) {
      return errors::InvalidArgument(
          "Mismatched number of replicas between executable and arguments for "
          " computation ",
          computation,
          ". Expected replicas=", device_assignment_.replica_count(),
          "; got replicas=", args[computation].n1());
    }
    for (int replica = 0; replica < device_assignment_.replica_count();
         ++replica) {
      int xrt_device_ordinal = device_assignment_(replica, computation);
      int tf_device_id = context_->tf_device_ids().at(xrt_device_ordinal);
      for (int arg = 0; arg < args[computation].n2(); ++arg) {
        const std::shared_ptr<XrtBuffer>& buffer =
            args[computation](replica, arg);
        if (buffer->handle().device_id() != tf_device_id) {
          return errors::InvalidArgument(
              "Input buffer to ExecuteReplicated() is not on the device for "
              "which the computation was compiled. Target device is ",
              tf_device_id, ", buffer is on device ",
              buffer->handle().device_id());
        }
      }
    }
  }

  std::vector<int> input_arity;
  input_arity.reserve(args.size());
  for (const auto& arg : args) {
    input_arity.push_back(arg.n2());
  }
  TF_ASSIGN_OR_RETURN(string exec_fn, context_->GetExecuteReplicatedFunction(
                                          input_arity, device_assignment_));

  std::vector<DataType> input_types;
  std::vector<const XrtTensorHandle*> inputs;
  inputs.push_back(&handle_);
  input_types.push_back(DT_INT64);

  std::vector<XrtTensorHandle> execution_config_handles(
      device_assignment_.computation_count());
  int tf_cpu_device_id = context_->tf_context()->cpu_device_id();
  for (int j = 0; j < device_assignment_.computation_count(); ++j) {
    TensorProto config_proto;
    config_proto.set_dtype(DT_STRING);
    xrt::XRTExecutionConfig config;
    config.set_core_index_in_replica(j);
    config_proto.add_string_val(config.SerializeAsString());
    execution_config_handles[j] = EnqueueConst(context_->tf_context().get(),
                                               tf_cpu_device_id, config_proto,
                                               /*host_memory=*/true);
    inputs.push_back(&execution_config_handles[j]);
    input_types.push_back(DT_STRING);
  }

  for (int i = 0; i < device_assignment_.replica_count(); ++i) {
    for (int j = 0; j < device_assignment_.computation_count(); ++j) {
      for (int k = 0; k < args[j].n2(); ++k) {
        inputs.push_back(&args[j](i, k)->handle());
        input_types.push_back(DT_INT64);
      }
    }
  }

  // Run all the XRTExecute ops in parallel using a multi-device function.
  // We do this for two reasons:
  // a) we need the operators to run in parallel, but without async mode enabled
  //    they might not.
  // b) we need the operators to all be issued as part of the same
  //    EnqueueRequest batch, otherwise we will deadlock.
  // TODO(phawkins): It would be even better to enable async mode, when its
  // error semantics have been improved.
  std::vector<DataType> output_types(device_assignment_.num_elements(),
                                     DT_INT64);
  std::vector<XrtTensorHandle> outputs = context_->tf_context()->EnqueueOp(
      exec_fn, inputs, /*output_arity=*/output_types.size(), /*attrs=*/{},
      tf_cpu_device_id);

  xla::Array2D<std::shared_ptr<XrtBuffer>> results(
      device_assignment_.computation_count(),
      device_assignment_.replica_count());
  int output_num = 0;
  for (int i = 0; i < device_assignment_.computation_count(); ++i) {
    for (int j = 0; j < device_assignment_.replica_count(); ++j) {
      int xrt_device_ordinal = device_assignment_(j, i);  // NB. different order
      int tf_device_id = context_->tf_device_ids().at(xrt_device_ordinal);

      // EnqueueOp doesn't know about multidevice functions, so it will assume
      // that the outputs are on the CPU. Override the device IDs it assigned;
      // we know better.
      outputs[output_num].set_device_id(tf_device_id);

      // TODO(phawkins): use a per-core result shape here.
      results(i, j) = std::make_shared<XrtBuffer>(
          std::move(outputs[output_num]), shape_.result());
      ++output_num;
    }
  }
  return results;
}

/*static*/ xla::StatusOr<std::shared_ptr<XrtContext>> XrtContext::Create(
    std::shared_ptr<XrtTfContext> tf_context, string device_type) {
  auto context = std::make_shared<XrtContext>(tf_context, device_type);
  if (context->tf_device_ids().empty()) {
    return errors::NotFound("No accelerator devices of type ", device_type,
                            " are present.");
  }
  if (device_type == "TPU") {
    TF_RETURN_IF_ERROR(context->InitializeTPU());
  } else {
    // Fill in a dummy topology mapping for CPU/GPU.
    for (int i = 0; i < context->tf_device_ids().size(); ++i) {
      context->device_mesh_coordinates_.push_back({});
      context->device_mesh_coordinates_.back().add_value(i);
    }
  }
  return context;
}

XrtContext::XrtContext(std::shared_ptr<XrtTfContext> tf_context,
                       string device_type)
    : tf_context_(std::move(tf_context)), device_type_(std::move(device_type)) {
  for (int i = 0; i < tf_context_->devices().size(); ++i) {
    const DeviceAttributes& device = tf_context_->devices()[i];
    VLOG(2) << "Device: " << i << ": " << device.DebugString();
    if (device.device_type() == device_type_) {
      tf_device_ids_.push_back(i);
      VLOG(1) << "Accelerator device " << i << ": " << device.name();
    }
  }
}

int XrtContext::device_count() const { return tf_device_ids_.size(); }

static Status RegisterTPUInitializeFunction(XrtTfContext* context) {
  FunctionDef fdef;
  OpDef* opdef = fdef.mutable_signature();
  opdef->set_name("TPUInitFunc");
  OpDef::ArgDef* outdef = opdef->add_output_arg();
  outdef->set_name("topology");
  outdef->set_type(DT_STRING);

  NodeDef* ndef = fdef.add_node_def();
  ndef->set_name("n");
  ndef->set_op("ConfigureDistributedTPU");

  (*fdef.mutable_ret())["topology"] = "n:topology";

  Status status = context->RegisterFunction(fdef);
  VLOG(10) << "RegisterTPUInitializeFunction returned " << status;
  return status;
}

Status XrtContext::InitializeTPU() {
  LOG(INFO) << "Initializing TPU devices.";
  TF_RETURN_IF_ERROR(RegisterTPUInitializeFunction(tf_context_.get()));

  TensorProto index_proto;
  index_proto.set_dtype(DT_INT32);
  index_proto.add_int_val(0);
  XrtTensorHandle device_ordinal = EnqueueConst(
      tf_context_.get(), /*device_id=*/tf_context_->cpu_device_id(),
      index_proto, /*host_memory=*/false);

  protobuf::Map<string, AttrValue> attrs;
  attrs["f"].mutable_func()->set_name("TPUInitFunc");
  attrs["Tin"].mutable_list();
  attrs["Tout"].mutable_list()->add_type(DT_STRING);
  XrtTensorHandle t = std::move(
      tf_context_->EnqueueOp("TPUPartitionedCall", {&device_ordinal},
                             /*output_arity=*/1,
                             /*attrs=*/attrs, tf_context_->cpu_device_id())[0]);

  auto result = tf_context_->RecvTensor(t, DT_STRING, /*host_memory=*/false);
  TF_ASSIGN_OR_RETURN(RecvTensorResponse * response, result->Get());
  VLOG(10) << "TPU topology " << response->DebugString();

  TF_ASSIGN_OR_RETURN(std::string data,
                      DeserializeTensorProtoAsString(response->tensor()));

  tpu::TopologyProto tpu_topology;
  tpu_topology.ParsePartialFromString(data);
  VLOG(4) << "TPU topology:\n" << tpu_topology.DebugString();

  TF_RET_CHECK(tpu_topology.num_tasks() == 1) << tpu_topology.DebugString();
  TF_RET_CHECK(tpu_topology.num_tpu_devices_per_task() == tf_device_ids_.size())
      << tpu_topology.DebugString() << " " << tf_device_ids_.size();

  const int mesh_rank = tpu_topology.mesh_shape_size();
  TF_RET_CHECK(tpu_topology.device_coordinates_size() ==
               tf_device_ids_.size() * mesh_rank);

  for (int i = 0; i < tf_device_ids_.size(); ++i) {
    device_mesh_coordinates_.push_back({});
    auto& coords = device_mesh_coordinates_.back();
    for (int j = 0; j < mesh_rank; ++j) {
      coords.add_value(tpu_topology.device_coordinates(i * mesh_rank + j));
    }
  }

  LOG(INFO) << "TPU initialization succeeded.";
  return Status::OK();
}

XrtContext::ExecuteReplicatedKey::ExecuteReplicatedKey(
    absl::Span<const int> input_arity, xla::DeviceAssignment device_assignment)
    : input_arity(input_arity.begin(), input_arity.end()),
      device_assignment(std::move(device_assignment)) {}

bool XrtContext::ExecuteReplicatedKey::operator==(
    const ExecuteReplicatedKey& other) const {
  return input_arity == other.input_arity &&
         device_assignment == other.device_assignment;
}

xla::StatusOr<string> XrtContext::GetExecuteReplicatedFunction(
    absl::Span<const int> input_arity,
    const xla::DeviceAssignment& device_assignment) {
  ExecuteReplicatedKey key(input_arity, device_assignment);

  absl::MutexLock lock(&mu_);
  auto it = replicated_fns_.find(key);
  if (it != replicated_fns_.end()) {
    return it->second;
  }

  string name = absl::StrCat("ExecuteReplicated_", replicated_fns_.size());

  FunctionDef fdef;
  OpDef* opdef = fdef.mutable_signature();
  opdef->set_name(name);
  OpDef::ArgDef* execution_handle = opdef->add_input_arg();
  execution_handle->set_name("execution_handle");
  execution_handle->set_type(DT_INT64);

  TF_RET_CHECK(device_assignment.computation_count() == input_arity.size());

  std::vector<OpDef::ArgDef*> execution_configs;
  execution_configs.reserve(device_assignment.computation_count());
  for (int j = 0; j < device_assignment.computation_count(); ++j) {
    OpDef::ArgDef* execution_config = opdef->add_input_arg();
    execution_config->set_name(absl::StrCat("execution_config_computation", j));
    execution_config->set_type(DT_STRING);
    execution_configs.push_back(execution_config);
  }

  for (int i = 0; i < device_assignment.replica_count(); ++i) {
    for (int j = 0; j < device_assignment.computation_count(); ++j) {
      NodeDef* ndef = fdef.add_node_def();
      ndef->set_name(absl::StrFormat("execute_replica%d_computation%d", i, j));
      ndef->set_op("XRTExecute");
      (*ndef->mutable_attr())["Ninputs"] = MakeAttrValue(input_arity[j]);
      ndef->add_input(execution_handle->name());
      ndef->add_input(execution_configs[j]->name());
      int tf_device_id = tf_device_ids_.at(device_assignment(i, j));
      ndef->set_device(tf_context_->devices().at(tf_device_id).name());

      for (int k = 0; k < input_arity[j]; ++k) {
        OpDef::ArgDef* arg = opdef->add_input_arg();
        arg->set_name(
            absl::StrFormat("in_replica%d_computation%d_arg%d", i, j, k));
        arg->set_type(DT_INT64);

        ndef->add_input(arg->name());
      }
      OpDef::ArgDef* ret = opdef->add_output_arg();
      ret->set_name(absl::StrFormat("out_replica%d_computation%d", i, j));
      ret->set_type(DT_INT64);

      (*fdef.mutable_ret())[ret->name()] =
          absl::StrCat(ndef->name(), ":output_handle");
    }
  }

  VLOG(10) << fdef.DebugString();

  Status status = tf_context_->RegisterFunction(fdef);
  VLOG(4) << "GetExecuteReplicatedFunction returned " << status;
  if (!status.ok()) return status;

  replicated_fns_[key] = name;
  return name;
}

}  // namespace tensorflow
