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

#include "tensorflow/compiler/xla/python/xrt.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_compile_ops.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_execute_op.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_state_ops.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace swig {

XrtAllocation::XrtAllocation(int64 handle, Shape shape,
                             const string& session_target)
    : handle_(handle), shape_(shape), session_target_(session_target) {}

XrtAllocation::~XrtAllocation() {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto allocation_handle =
      tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto release =
      tensorflow::ops::XRTReleaseAllocationHandle(root, allocation_handle);
  if (!root.status().ok()) {
    LOG(ERROR) << root.status();
    return;
  }

  tensorflow::ClientSession session(root, session_target_);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({allocation_handle, handle()});
  std::vector<tensorflow::Tensor> outputs;
  auto status = session.Run(inputs, {}, {release}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return;
  }
}

/* static */
StatusOr<XrtAllocation*> XrtAllocation::FromLiteral(
    const Literal& argument, const string& session_target) {
  xrt::XLAAllocation alloc;
  *alloc.mutable_value() = argument.ToProto();

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto literal_string =
      tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
  auto literal_handle = tensorflow::ops::XRTAllocate(root, literal_string);
  TF_RETURN_IF_ERROR(root.status());

  tensorflow::ClientSession session(root, session_target);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({literal_string, alloc.SerializeAsString()});
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session.Run(inputs, {literal_handle}, &outputs));

  int64 handle = outputs[0].scalar<int64>()();
  return new XrtAllocation(handle, argument.shape(), session_target);
}

const int64 XrtAllocation::handle() const { return handle_; }

const Shape& XrtAllocation::shape() const { return shape_; }

StatusOr<Literal> XrtAllocation::ToLiteral() const {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto allocation_handle =
      tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto read_literal = tensorflow::ops::XRTReadLiteral(root, allocation_handle);
  TF_RETURN_IF_ERROR(root.status());

  tensorflow::ClientSession session(root, session_target_);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({allocation_handle, handle()});
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session.Run(inputs, {read_literal}, &outputs));

  xla::LiteralProto response;
  TF_RET_CHECK(response.ParseFromString(outputs[0].scalar<string>()()));
  return Literal::CreateFromProto(response);
}

XrtAllocationTuple::XrtAllocationTuple(std::vector<XrtAllocation*> elements)
    : elements_(std::move(elements)) {
  for (auto* element : elements_) {
    CHECK(element != nullptr);
  }
}

XrtAllocationTuple::~XrtAllocationTuple() {
  for (XrtAllocation* element : elements_) {
    if (element != nullptr) {
      delete element;
    }
  }
}

StatusOr<XrtAllocation*> XrtAllocationTuple::Release(int i) {
  XrtAllocation* element = elements_[i];
  if (element == nullptr) {
    return InvalidArgument("Attempted to release already-released element %d.",
                           i);
  }
  elements_[i] = nullptr;
  return element;
}

int64 XrtAllocationTuple::size() const { return elements_.size(); }

StatusOr<XrtExecutable*> XrtExecutable::CompileForXrt(
    const string& hlo_module_proto, const std::vector<Shape>& argument_shapes,
    const Shape& result_shape, const string& session_target) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto program = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
  auto compile = tensorflow::ops::XRTCompile(root, program);
  TF_RETURN_IF_ERROR(root.status());

  xrt::XLAComputation c;
  auto config = c.mutable_config();
  ProgramShape program_shape;
  for (auto& shape : argument_shapes) {
    *program_shape.add_parameters() = shape;
  }
  *program_shape.mutable_result() = result_shape;

  LayoutUtil::SetToDefaultLayout(&program_shape);
  *config->mutable_program_shape() = program_shape.ToProto();
  c.mutable_hlo_snapshot()
      ->mutable_hlo()
      ->mutable_hlo_module()
      ->ParsePartialFromString(hlo_module_proto);

  tensorflow::ClientSession session(root, session_target);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({program, c.SerializeAsString()});
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session.Run(inputs, {compile.handle}, &outputs));

  int64 handle = outputs[0].scalar<int64>()();
  return new XrtExecutable(program_shape, handle, session_target);
}

XrtExecutable::XrtExecutable(const ProgramShape& program_shape, int64 handle,
                             const string& session_target)
    : program_shape_(program_shape),
      handle_(handle),
      session_target_(session_target) {}

XrtExecutable::~XrtExecutable() {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto computation_handle =
      tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto release =
      tensorflow::ops::XRTReleaseCompilationHandle(root, computation_handle);
  if (!root.status().ok()) {
    LOG(ERROR) << root.status();
    return;
  }

  tensorflow::ClientSession session(root, session_target_);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({computation_handle, handle()});
  std::vector<tensorflow::Tensor> outputs;
  auto status = session.Run(inputs, {}, {release}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return;
  }
}

StatusOr<XrtAllocation*> XrtExecutable::Execute(
    absl::Span<XrtAllocation* const> argument_handles) {
  const int num_expected_arguments = program_shape().parameters().size();

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  std::vector<tensorflow::Output> arguments;
  arguments.reserve(num_expected_arguments);
  for (int i = 0; i < num_expected_arguments; ++i) {
    arguments.push_back(
        tensorflow::ops::Placeholder(root, tensorflow::DT_INT64));
  }
  auto computation_handle =
      tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto execution_config =
      tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
  auto execute = tensorflow::ops::XRTExecute(root, computation_handle,
                                             execution_config, arguments);
  TF_RETURN_IF_ERROR(root.status());

  TF_RET_CHECK(argument_handles.size() == arguments.size());

  xrt::XRTExecutionConfig e;
  e.set_release_input_handles(false);
  e.set_release_compilation_handle(false);

  tensorflow::ClientSession session(root, session_target_);
  tensorflow::ClientSession::FeedType inputs;
  for (int i = 0; i < arguments.size(); ++i) {
    inputs.insert({arguments[i], argument_handles[i]->handle()});
  }
  inputs.insert({computation_handle, handle()});
  inputs.insert({execution_config, e.SerializeAsString()});
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session.Run(inputs, {execute}, &outputs));

  int64 output = outputs[0].scalar<int64>()();
  return new XrtAllocation(output, program_shape().result(), session_target_);
}

const ProgramShape& XrtExecutable::program_shape() const {
  return program_shape_;
}

int64 XrtExecutable::handle() const { return handle_; }

void DeleteXrtAllocation(XrtAllocation* allocation) { delete allocation; }

void DeleteXrtExecutable(XrtExecutable* computation) { delete computation; }

StatusOr<XrtAllocationTuple*> DestructureXrtAllocationTuple(
    XrtAllocation* allocation, const string& session_target) {
  const Shape& tuple_shape = allocation->shape();

  if (!tuple_shape.IsTuple()) {
    return InvalidArgument(
        "Attemped to destructure a LocalShapedBuffer that did not have a tuple "
        "shape; shape: %s",
        ShapeUtil::HumanString(tuple_shape));
  }

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto base_handle = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto shape_index = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
  auto subtuple = tensorflow::ops::XRTSubTuple(root, base_handle, shape_index);
  TF_RETURN_IF_ERROR(root.status());

  tensorflow::ClientSession session(root, session_target);
  tensorflow::ClientSession::FeedType inputs;
  std::vector<XrtAllocation*> results;
  for (int32 i = 0; i < ShapeUtil::TupleElementCount(tuple_shape); ++i) {
    inputs.clear();
    inputs.insert({base_handle, allocation->handle()});
    inputs.insert({shape_index, {i}});
    std::vector<tensorflow::Tensor> outputs;
    auto status = session.Run(inputs, {subtuple}, &outputs);
    if (!status.ok()) {
      // Clean up before returning non-ok status.
      for (int j = 0; j < results.size(); ++j) {
        delete results[j];
      }
      return status;
    }
    const int64 subtuple_handle = outputs[0].scalar<int64>()();
    const Shape& subtuple_shape =
        ShapeUtil::GetTupleElementShape(tuple_shape, i);
    results.push_back(
        new XrtAllocation(subtuple_handle, subtuple_shape, session_target));
  }
  return new XrtAllocationTuple(std::move(results));
}

}  // namespace swig
}  // namespace xla
