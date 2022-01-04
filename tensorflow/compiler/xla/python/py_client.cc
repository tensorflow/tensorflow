/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/py_client.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/pjrt/transpose.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/core/profiler/profile.pb.h"

namespace xla {

namespace py = pybind11;
namespace pprof = tensorflow::tfprof::pprof;

PyClient::PyClient(std::unique_ptr<PjRtClient> pjrt_client)
    : PyClient(std::shared_ptr<PjRtClient>(std::move(pjrt_client))) {}

PyClient::PyClient(std::shared_ptr<PjRtClient> pjrt_client)
    : pjrt_client_(std::move(pjrt_client)) {
  CHECK(pjrt_client_ != nullptr);
  buffers_.resize(pjrt_client_->device_count());
  for (PjRtDevice* device : pjrt_client_->addressable_devices()) {
    if (device->id() >= buffers_.size()) {
      buffers_.resize(device->id() + 1);
    }
  }
}

PyClient::~PyClient() {
  py::gil_scoped_release gil;
  pjrt_client_ = nullptr;
}

std::vector<ClientAndPtr<PjRtDevice>> PyClient::Devices() {
  std::vector<ClientAndPtr<PjRtDevice>> devices;
  auto span = pjrt_client_->devices();
  devices.reserve(span.size());
  for (PjRtDevice* device : span) {
    devices.push_back(WrapWithClient(shared_from_this(), device));
  }
  return devices;
}

std::vector<ClientAndPtr<PjRtDevice>> PyClient::LocalDevices() {
  std::vector<ClientAndPtr<PjRtDevice>> devices;
  devices.reserve(pjrt_client_->addressable_devices().size());
  for (PjRtDevice* device : pjrt_client_->addressable_devices()) {
    devices.push_back(WrapWithClient(shared_from_this(), device));
  }
  return devices;
}

std::vector<py::object> PyClient::LiveBuffers() {
  CHECK(PyGILState_Check());
  std::vector<py::object> buffers;
  for (PyBuffer* device_buffers : buffers_) {
    for (PyBuffer* buffer = device_buffers; buffer; buffer = buffer->next_) {
      if (!buffer->is_deleted()) {
        buffers.push_back(
            py::reinterpret_borrow<py::object>(buffer->AsHandle()));
      }
    }
  }
  return buffers;
}

std::vector<py::object> PyClient::LiveBuffersOnDevice(PjRtDevice* device) {
  CHECK_EQ(device->client(), pjrt_client());
  CHECK(PyGILState_Check());
  std::vector<py::object> buffers;
  for (PyBuffer* buffer = buffers_[device->id()]; buffer;
       buffer = buffer->next_) {
    if (!buffer->is_deleted()) {
      buffers.push_back(py::reinterpret_borrow<py::object>(buffer->AsHandle()));
    }
  }
  return buffers;
}

std::vector<std::shared_ptr<PyExecutable>> PyClient::LiveExecutables() {
  CHECK(PyGILState_Check());
  std::vector<std::shared_ptr<PyExecutable>> executables;
  for (PyExecutable* exec = executables_; exec; exec = exec->next_) {
    if (!exec->is_deleted()) {
      executables.push_back(exec->shared_from_this());
    }
  }
  return executables;
}

Status PyClient::Defragment() {
  CHECK(PyGILState_Check());
  switch (pjrt_client_->runtime_type()) {
    case PjRtRuntimeType::kTfrt:
      return pjrt_client_->Defragment();
    case PjRtRuntimeType::kStreamExecutor:
      struct TmpBuffer {
        PyBuffer* py_buffer;
        // TODO(skyewm): maybe use py_buffer's HostValue
        std::shared_ptr<Literal> host_copy;
      };

      // Synchronously copy all buffers to host
      std::vector<TmpBuffer> tmp_buffers;
      for (PyBuffer* device_buffers : buffers_) {
        for (PyBuffer* buffer = device_buffers; buffer;
             buffer = buffer->next_) {
          if (!buffer->is_deleted()) {
            TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal,
                                buffer->buffer_->ToLiteral());
            tmp_buffers.push_back({buffer, literal});
          }
        }
      }

      // All buffers successfully copied to host, delete on-device copies.
      //
      // Use blocking delete operation to ensure all memory is actually cleared
      // before we start rewriting buffers.
      //
      // Die instead of returning a bad status because program presumably can't
      // continue if we fail to reconstitute device buffers.
      for (TmpBuffer& tmp_buffer : tmp_buffers) {
        TF_CHECK_OK(tensorflow::down_cast<PjRtStreamExecutorBuffer*>(
                        tmp_buffer.py_buffer->buffer_.get())
                        ->Release(/*wait_for_operations_to_complete=*/true)
                        .status());
      }

      // Copy host copies back to device and update PyBuffers in-place.
      for (TmpBuffer& tmp_buffer : tmp_buffers) {
        std::unique_ptr<PjRtBuffer> new_copy =
            pjrt_client_
                ->BufferFromHostLiteral(*tmp_buffer.host_copy,
                                        tmp_buffer.py_buffer->buffer_->device())
                .ValueOrDie();
        TF_CHECK_OK(new_copy->BlockHostUntilReady());
        tmp_buffer.py_buffer->buffer_.reset(new_copy.release());
      }

      // TODO(skyewm): delete executables?
  }
  return Status::OK();
}

StatusOr<std::vector<std::vector<ClientAndPtr<PjRtDevice>>>>
PyClient::GetDefaultDeviceAssignment(int num_replicas, int num_partitions) {
  TF_ASSIGN_OR_RETURN(
      DeviceAssignment device_assignment,
      pjrt_client_->GetDefaultDeviceAssignment(num_replicas, num_partitions));
  std::vector<std::vector<ClientAndPtr<PjRtDevice>>> result;
  result.resize(num_replicas);
  for (int r = 0; r < num_replicas; ++r) {
    result[r].resize(num_partitions);
    for (int p = 0; p < num_partitions; ++p) {
      int device_id = device_assignment(r, p);
      TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                          pjrt_client_->LookupDevice(device_id));
      result[r][p] = WrapWithClient(shared_from_this(), device);
    }
  }
  return result;
}

StatusOr<std::vector<ClientAndPtr<PjRtDevice>>>
PyClient::GetDefaultDeviceAssignment1D(int num_replicas) {
  TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                      pjrt_client_->GetDefaultDeviceAssignment(
                          num_replicas, /*num_partitions=*/1));
  std::vector<ClientAndPtr<PjRtDevice>> result;
  for (int i = 0; i < num_replicas; ++i) {
    int device_id = device_assignment(i, 0);
    TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                        pjrt_client_->LookupDevice(device_id));
    result.push_back(WrapWithClient(shared_from_this(), device));
  }
  return result;
}

StatusOr<py::object> PyClient::BufferFromPyval(
    pybind11::handle argument, PjRtDevice* device, bool force_copy,
    PjRtClient::HostBufferSemantics host_buffer_semantics) {
  if (device == nullptr) {
    TF_RET_CHECK(!pjrt_client_->addressable_devices().empty());
    device = pjrt_client_->addressable_devices().front();
  }
  CHECK(device != nullptr);
  TF_ASSIGN_OR_RETURN(PjRtDevice * found_device,
                      pjrt_client_->LookupDevice(device->id()));
  if (found_device != device) {
    return InvalidArgument("Cannot copy value to device '%s' with '%s' backend",
                           device->DebugString(),
                           pjrt_client_->platform_name());
  }
  GlobalPyRefManager()->CollectGarbage();

  DevicePutOptions options;
  options.squash_64bit_types = false;
  options.allow_zero_copy =
      (!force_copy &&
       (host_buffer_semantics == PjRtClient::HostBufferSemantics::kZeroCopy));
  options.force_lazy_arrays = true;
  TF_ASSIGN_OR_RETURN(DevicePutResult put,
                      DevicePut(argument, device, options));

  if (put.owned_buffer) {
    auto traceback = Traceback::Get();
    return PyBuffer::Make(shared_from_this(), std::move(put.owned_buffer),
                          std::move(traceback));
  } else {
    return py::reinterpret_borrow<py::object>(put.owning_pybuffer);
  }
}

StatusOr<std::shared_ptr<PyExecutable>> PyClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  std::unique_ptr<PjRtExecutable> executable;
  absl::optional<std::string> fingerprint;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(executable,
                        pjrt_client_->Compile(computation, std::move(options)));
    TF_ASSIGN_OR_RETURN(fingerprint,
                        pjrt_client_->ExecutableFingerprint(*executable));
  }
  auto traceback = Traceback::Get();
  return std::make_shared<PyExecutable>(
      shared_from_this(), std::move(executable), std::move(traceback),
      std::move(fingerprint));
}

StatusOr<std::shared_ptr<PyExecutable>> PyClient::CompileMlir(
    absl::string_view mlir_module, CompileOptions options) {
  std::unique_ptr<PjRtExecutable> executable;
  absl::optional<std::string> fingerprint;
  {
    py::gil_scoped_release gil_release;
    mlir::MLIRContext context;
    TF_ASSIGN_OR_RETURN(mlir::OwningModuleRef module,
                        ParseMlirModuleString(mlir_module, context));
    TF_ASSIGN_OR_RETURN(
        executable, pjrt_client_->Compile(module.get(), std::move(options)));
    TF_ASSIGN_OR_RETURN(fingerprint,
                        pjrt_client_->ExecutableFingerprint(*executable));
  }
  auto traceback = Traceback::Get();
  return std::make_shared<PyExecutable>(
      shared_from_this(), std::move(executable), std::move(traceback),
      std::move(fingerprint));
}

StatusOr<py::bytes> PyClient::SerializeExecutable(
    const PyExecutable& executable) const {
  return pjrt_client_->SerializeExecutable(executable.pjrt_executable());
}

StatusOr<std::shared_ptr<PyExecutable>> PyClient::DeserializeExecutable(
    const std::string& serialized, CompileOptions options) {
  std::unique_ptr<PjRtExecutable> executable;
  absl::optional<std::string> fingerprint;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(executable, pjrt_client_->DeserializeExecutable(
                                        serialized, std::move(options)));
    TF_ASSIGN_OR_RETURN(fingerprint,
                        pjrt_client_->ExecutableFingerprint(*executable));
  }
  auto traceback = Traceback::Get();
  return std::make_shared<PyExecutable>(
      shared_from_this(), std::move(executable), std::move(traceback),
      std::move(fingerprint));
}

class ProfileBuilder {
 public:
  ProfileBuilder();
  pprof::Profile& profile() { return profile_; }

  // Adds or returns the ID of `s` in the table.
  int StringId(const std::string& s);

  // Adds or returns the ID of a function.
  int FunctionId(PyCodeObject* code);

  // Adds or returns the ID of a code location.
  int LocationId(PyCodeObject* code, int instruction);

 private:
  pprof::Profile profile_;

  absl::flat_hash_map<std::string, int> strings_;
  absl::flat_hash_map<PyCodeObject*, int> functions_;
  absl::flat_hash_map<std::pair<PyCodeObject*, int>, int> locations_;
};

ProfileBuilder::ProfileBuilder() { CHECK_EQ(0, StringId("")); }

int ProfileBuilder::StringId(const std::string& s) {
  auto ret = strings_.emplace(s, profile_.string_table_size());
  if (ret.second) {
    profile_.add_string_table(s);
  }
  return ret.first->second;
}

int ProfileBuilder::FunctionId(PyCodeObject* code) {
  // +1 because id 0 is reserved.
  auto ret = functions_.emplace(code, profile_.function_size() + 1);
  if (ret.second) {
    auto* function = profile_.add_function();
    function->set_id(ret.first->second);
    int name = StringId(py::str(code->co_name));
    function->set_name(name);
    function->set_system_name(name);
    function->set_filename(StringId(py::str(code->co_filename)));
    function->set_start_line(code->co_firstlineno);
  }
  return ret.first->second;
}

int ProfileBuilder::LocationId(PyCodeObject* code, int instruction) {
  // +1 because id 0 is reserved.
  auto ret = locations_.emplace(std::make_pair(code, instruction),
                                profile_.location_size() + 1);
  if (ret.second) {
    auto* location = profile_.add_location();
    location->set_id(ret.first->second);
    auto* line = location->add_line();
    line->set_function_id(FunctionId(code));
    line->set_line(PyCode_Addr2Line(code, instruction));
  }
  return ret.first->second;
}

namespace {

struct HeapProfileKey {
  Traceback* traceback;
  int64_t size;
  PjRtDevice* device;
  bool operator==(const HeapProfileKey& other) const;
};

bool HeapProfileKey::operator==(const HeapProfileKey& other) const {
  if (size != other.size || device != other.device) {
    return false;
  }
  if ((traceback == nullptr) != (other.traceback == nullptr)) {
    return false;
  }
  if (traceback && traceback->raw_frames() != other.traceback->raw_frames()) {
    return false;
  }
  return true;
}

template <typename H>
H AbslHashValue(H h, const HeapProfileKey& key) {
  if (key.traceback) {
    h = H::combine_contiguous(std::move(h), key.traceback->raw_frames().begin(),
                              key.traceback->raw_frames().size());
  }
  h = H::combine(std::move(h), key.size, key.device);
  return h;
}

}  // namespace

StatusOr<py::bytes> PyClient::HeapProfile() {
  CHECK(PyGILState_Check());
  absl::flat_hash_set<PjRtBuffer*> buffer_set;
  absl::flat_hash_map<HeapProfileKey, int64_t> entries;
  for (PyBuffer* device_buffers : buffers_) {
    for (PyBuffer* buffer = device_buffers; buffer; buffer = buffer->next_) {
      // We only wish to count each PjRtBuffer once, even though they may be
      // shared by multiple PyBuffers.
      if (!buffer->is_deleted() && buffer_set.insert(buffer->buffer()).second) {
        TF_ASSIGN_OR_RETURN(size_t size,
                            buffer->buffer()->GetOnDeviceSizeInBytes());
        HeapProfileKey key{buffer->traceback().get(),
                           static_cast<int64_t>(size),
                           buffer->buffer()->device()};
        ++entries[key];
      }
    }
  }

  for (PyExecutable* executable = executables_; executable;
       executable = executable->next_) {
    if (!executable->is_deleted()) {
      HeapProfileKey key{executable->traceback(),
                         executable->SizeOfGeneratedCodeInBytes(), nullptr};
      ++entries[key];
    }
  }

  ProfileBuilder builder;
  auto* allocations = builder.profile().add_sample_type();
  allocations->set_type(builder.StringId("allocations"));
  allocations->set_unit(builder.StringId("count"));
  auto* space = builder.profile().add_sample_type();
  space->set_type(builder.StringId("space"));
  space->set_unit(builder.StringId("bytes"));

  const int kind_string_id = builder.StringId("kind");
  const int buffer_string_id = builder.StringId("buffer");
  const int executable_string_id = builder.StringId("executable");
  const int device_string_id = builder.StringId("device");
  for (const auto& entry : entries) {
    auto* sample = builder.profile().add_sample();
    if (entry.first.traceback) {
      for (const auto& frame : entry.first.traceback->raw_frames()) {
        sample->add_location_id(builder.LocationId(frame.first, frame.second));
      }
    }
    sample->add_value(entry.second);
    sample->add_value(entry.first.size * entry.second);

    auto* kind_label = sample->add_label();
    kind_label->set_key(kind_string_id);
    if (entry.first.device) {
      kind_label->set_str(buffer_string_id);
      auto* device_label = sample->add_label();
      device_label->set_key(device_string_id);
      device_label->set_str(
          builder.StringId(entry.first.device->DebugString()));
    } else {
      kind_label->set_str(executable_string_id);
    }
  }
  return py::bytes(builder.profile().SerializeAsString());
}

namespace {

class CpuCallback {
 public:
  struct Arg {
    PrimitiveType type;                    // XLA type
    py::dtype dtype;                       // NumPy type, for array types.
    absl::InlinedVector<int64_t, 4> dims;  // Dimensions, for array types.
    std::vector<ssize_t> strides;          // Byte strides, for array types.
  };
  struct Result {
    PrimitiveType type;  // XLA type
    // Expected output shape, for array types
    absl::InlinedVector<int64_t, 4> expected_dims;
    // Expected output byte strides, for array types. If the strides do not
    // match the output will be transposed into the expected layout.
    std::vector<int64_t> expected_strides;
    // The desired order of output dimensions in major-to-minor order.
    absl::InlinedVector<int64_t, 4> reversed_layout;
    // Size of the array in bytes.
    size_t size_in_bytes;
  };

  explicit CpuCallback(py::function callable, std::vector<Arg> args,
                       std::vector<Result> results)
      : callable_(std::move(callable)),
        args_(std::move(args)),
        results_(std::move(results)),
        transpose_cache_(/*capacity=*/16) {}

  void Call(void* result, void** arg_ptrs);

 private:
  py::function callable_;
  std::vector<Arg> const args_;
  std::vector<Result> const results_;
  TransposePlanCache transpose_cache_;
};

void CpuCallback::Call(void* result, void** arg_ptrs) {
  absl::Span<void* const> inputs(arg_ptrs, args_.size());
  absl::Span<void* const> outputs(reinterpret_cast<void**>(result),
                                  results_.size());

  py::gil_scoped_acquire gil;
  py::tuple args(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (args_[i].type == TOKEN) {
      args[i] = py::none();
    } else {
      args[i] = py::array(args_[i].dtype, args_[i].dims, args_[i].strides,
                          const_cast<void*>(inputs[i]));
      args[i].attr("flags").attr("writeable") = Py_False;
    }
  }
  py::object result_tuple = callable_(*py::reinterpret_borrow<py::args>(args));
  if (!PyTuple_Check(result_tuple.ptr())) {
    throw std::runtime_error(
        absl::StrFormat("CPU callback expected a tuple result, got %s",
                        static_cast<std::string>(py::repr(result_tuple))));
  }
  if (PyTuple_Size(result_tuple.ptr()) != results_.size()) {
    throw std::runtime_error(
        absl::StrFormat("CPU callback expected a tuple with %d results, got %d",
                        results_.size(), PyTuple_Size(result_tuple.ptr())));
  }
  for (size_t i = 0; i < results_.size(); ++i) {
    py::object output = py::reinterpret_borrow<py::object>(
        PyTuple_GetItem(result_tuple.ptr(), i));
    if (results_[i].type == TOKEN) {
      if (!output.is_none()) {
        throw std::runtime_error(absl::StrFormat(
            "Token output from Python callback should be None, got %s",
            static_cast<std::string>(py::repr(output))));
      }
      continue;
    }
    py::array array = py::cast<py::array>(std::move(output));
    static_assert(sizeof(ssize_t) == sizeof(int64_t),
                  "Expected ssize_t to be of equal size to int64_t");
    absl::Span<int64_t const> dims(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    if (dims != results_[i].expected_dims) {
      throw std::runtime_error(absl::StrFormat(
          "Mismatched result shape for %d-th return value from CPU callback; "
          "expected array with dimensions %s, got %s",
          i, absl::StrJoin(results_[i].expected_dims, ","),
          absl::StrJoin(dims, ",")));
    }
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    if (strides == results_[i].expected_strides) {
      std::memcpy(outputs[i], array.data(), results_[i].size_in_bytes);
    } else {
      StatusOr<std::shared_ptr<TransposePlan>> plan =
          transpose_cache_.GetOrCreate(
              primitive_util::ByteWidth(results_[i].type), dims,
              results_[i].reversed_layout,
              /*input_layout=*/TransposePlan::Striding{strides});
      if (!plan.ok()) {
        throw std::runtime_error(plan.status().ToString());
      }
      plan.ValueOrDie()->Execute(array.data(), outputs[i]);
    }
  }
}

extern "C" void XlaPythonCpuCallback(void* output, void** inputs) {
  CpuCallback* callback =
      absl::bit_cast<CpuCallback*>(*static_cast<uintptr_t*>(inputs[0]));
  callback->Call(output, inputs + 1);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("xla_python_cpu_callback",
                                             &XlaPythonCpuCallback);

}  // namespace

StatusOr<std::pair<XlaOp, pybind11::object>> PyClient::EmitPythonCallback(
    pybind11::function callable, XlaBuilder& builder,
    absl::Span<XlaOp const> operands, absl::Span<Shape const> result_shapes,
    absl::optional<std::vector<Shape>> operand_layouts, bool has_side_effect) {
  if (pjrt_client_->platform_id() != CpuId()) {
    return Unimplemented("EmitPythonCallback is only implemented on CPU");
  }

  std::vector<CpuCallback::Arg> callback_args(operands.size());
  std::vector<XlaOp> custom_call_args(operands.size() + 1);
  absl::c_copy(operands, custom_call_args.begin() + 1);

  if (operand_layouts && operand_layouts->size() != operands.size()) {
    return InvalidArgument(
        "Mismatched number of operands (%d) and operand_layouts (%d)",
        operands.size(), operand_layouts->size());
  }

  std::vector<Shape> custom_call_arg_layouts(operands.size() + 1);
  static_assert(sizeof(uintptr_t) == sizeof(uint64_t),
                "Expected 64-bit pointers");
  custom_call_arg_layouts[0] =
      ShapeUtil::MakeShapeWithDescendingLayout(U64, {});
  for (int i = 0; i < operands.size(); ++i) {
    TF_ASSIGN_OR_RETURN(Shape shape, builder.GetShape(operands[i]));
    xla::Shape& layout = custom_call_arg_layouts[i + 1];
    if (operand_layouts) {
      if (!(*operand_layouts)[i].has_layout()) {
        return InvalidArgument(
            "operand_layout shapes for callback must have "
            "layouts, got %s",
            (*operand_layouts)[i].ToString(/*print_layout=*/true));
      }
      if (!ShapeUtil::Compatible(shape, (*operand_layouts)[i])) {
        return InvalidArgument(
            "Incompatible shapes for Python callback argument %d: %s vs %s", i,
            shape.ToString(),
            (*operand_layouts)[i].ToString(/*print_layout=*/true));
      }
      layout = (*operand_layouts)[i];
    } else {
      layout = LayoutUtil::GetWithDefaultLayout(shape);
    }

    if (shape.IsArray()) {
      callback_args[i].dims.resize(shape.dimensions_size());
      absl::c_copy(shape.dimensions(), callback_args[i].dims.begin());
      callback_args[i].strides = ByteStridesForShape(layout);
      callback_args[i].type = shape.element_type();
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

  std::vector<Shape> result_shapes_with_layout(result_shapes.size());
  std::vector<CpuCallback::Result> callback_results(result_shapes.size());
  for (int i = 0; i < result_shapes.size(); ++i) {
    if (result_shapes[i].IsArray()) {
      result_shapes_with_layout[i] =
          result_shapes[i].has_layout()
              ? result_shapes[i]
              : LayoutUtil::GetWithDefaultLayout(result_shapes[i]);
      const Shape& shape = result_shapes_with_layout[i];
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
      result_shapes_with_layout[i] = result_shapes[i];
    } else {
      return InvalidArgument(
          "Only array and token return values from Python callbacks are "
          "supported, got %s",
          result_shapes[i].ToString());
    }
  }

  auto callback = std::make_unique<CpuCallback>(
      std::move(callable), callback_args, callback_results);
  custom_call_args[0] = ConstantR0<std::uint64_t>(
      &builder, absl::bit_cast<std::uint64_t>(callback.get()));

  Shape result_shape = ShapeUtil::MakeTupleShape(result_shapes_with_layout);
  XlaOp result = CustomCallWithLayout(&builder, "xla_python_cpu_callback",
                                      custom_call_args, result_shape,
                                      custom_call_arg_layouts,
                                      /*opaque=*/"", has_side_effect);

  py::capsule callback_capsule(callback.release(), [](void* ptr) {
    delete reinterpret_cast<CpuCallback*>(ptr);
  });
  return std::make_pair(result, py::object(std::move(callback_capsule)));
}

}  // namespace xla
