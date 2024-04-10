/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PY_CLIENT_H_
#define XLA_PYTHON_PY_CLIENT_H_

#include <Python.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/client/xla_builder.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/shape.h"

namespace xla {

class PyClient;
class PyLoadedExecutable;
class PyArray;
class PyDevice;
class PyMemorySpace;
struct PyArray_Storage;

// Python wrapper around PjRtClient.
// We use a wrapper class to add Python-specific functionality.
class PyClient {
 public:
  static nb_class_ptr<PyClient> Make(std::shared_ptr<ifrt::Client> ifrt_client);

  // Do not call the constructor directly. Use `PyClient::Make` instead.
  explicit PyClient(std::shared_ptr<ifrt::Client> ifrt_client);
  virtual ~PyClient();

  ifrt::Client* ifrt_client() const { return ifrt_client_.get(); }

  // Short-term escape hatch to get PjRtClient from PyClient.
  // TODO(hyeontaek): Migrate all users of this method to be agnostic of PjRt.
  xla::PjRtClient* pjrt_client() const {
    auto* pjrt_client =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(ifrt_client_.get());
    if (pjrt_client == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return pjrt_client->pjrt_client();
  }
  std::shared_ptr<PjRtClient> shared_ptr_pjrt_client() {
    auto* pjrt_client =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(ifrt_client_.get());
    if (pjrt_client == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return pjrt_client->shared_ptr_pjrt_client();
  }

  // Legacy alises.
  std::shared_ptr<PjRtClient> shared_pjrt_client() {
    return shared_ptr_pjrt_client();
  }

  std::string_view platform_name() const {
    // TODO(phawkins): this is a temporary backwards compatibility shim. We
    // changed the name PJRT reports for GPU platforms to "cuda" or "rocm", but
    // we haven't yet updated JAX clients that expect "gpu". Migrate users and
    // remove this code.
    if (ifrt_client_->platform_name() == "cuda" ||
        ifrt_client_->platform_name() == "rocm") {
      return "gpu";
    } else {
      return ifrt_client_->platform_name();
    }
  }
  std::string_view platform_version() const {
    return ifrt_client_->platform_version();
  }
  std::string_view runtime_type() const { return ifrt_client_->runtime_type(); }

  // Returns implementation-specific attributes about this client, e.g. the PJRT
  // C API version if applicable.
  absl::flat_hash_map<std::string, xla::ifrt::Client::ClientAttribute>
  attributes() const {
    return client_attributes_;
  }

  int addressable_device_count() const {
    return ifrt_client_->addressable_device_count();
  }
  int device_count() const { return ifrt_client_->device_count(); }
  int process_index() const { return ifrt_client_->process_index(); }

  std::vector<nb_class_ptr<PyDevice>> Devices();
  std::vector<nb_class_ptr<PyDevice>> LocalDevices();
  absl::StatusOr<nb_class_ptr<PyDevice>> DeviceFromLocalHardwareId(
      int local_hardware_id);

  // Returns the PyDevice associated with the given PjRtDevice.
  nb_class_ptr<PyDevice> GetPyDevice(PjRtDevice* device);

  // Returns the PyMemorySpace associated with the given PjRtMemorySpace.
  nb_class_ptr<PyMemorySpace> GetPyMemorySpace(PjRtMemorySpace* memory_space);

  // Returns a vector of live PyArray objects. PyArray objects may share
  // PjRtBuffers, so there may be duplicates of the same underlying device
  // buffer.
  std::vector<nanobind::object> LiveBuffersOnDevice(PjRtDevice* device);

  nanobind::list LiveExecutables();

  // TODO(zhangqiaorjc): Remove when we have transparent defragmentation.
  absl::Status Defragment();

  static absl::StatusOr<nanobind::object> BufferFromPyval(
      nb_class_ptr<PyClient> client, nanobind::handle argument,
      PjRtDevice* device, bool force_copy,
      ifrt::Client::HostBufferSemantics host_buffer_semantics);

  static absl::StatusOr<nb_class_ptr<PyLoadedExecutable>> CompileIfrtProgram(
      nb_class_ptr<PyClient> client,
      std::unique_ptr<ifrt::Program> ifrt_program,
      std::unique_ptr<ifrt::CompileOptions> ifrt_options);

  static absl::StatusOr<nb_class_ptr<PyLoadedExecutable>> Compile(
      nb_class_ptr<PyClient> client, std::string mlir_module,
      CompileOptions options, std::vector<nanobind::capsule> host_callbacks);

  absl::StatusOr<nanobind::bytes> SerializeExecutable(
      const PyLoadedExecutable& executable) const;
  static absl::StatusOr<nb_class_ptr<PyLoadedExecutable>> DeserializeExecutable(
      nb_class_ptr<PyClient> client, nanobind::bytes serialized,
      std::optional<CompileOptions> options,
      std::vector<nanobind::capsule> host_callbacks);

  absl::StatusOr<nanobind::bytes> HeapProfile();

  // `GetEmitPythonCallbackDescriptor` takes in an input Python callable that
  // takes in arguments of shapes `operand_shapes` and returns values of shapes
  // `result_shapes`. It returns a pair of a `uint64_t` descriptor and a Python
  // object whose reference will keep the Python callback alive. The descriptor
  // should be passed into a 'xla_python_cpu_callback' or
  // 'xla_python_gpu_callback' CustomCall as its first argument. Typically the
  // callback may be kept alive by attaching the keep-alive object to the
  // executable built from this computation.
  //
  // The callable receives as arguments NumPy arrays for arguments with array
  // types, and None for Token argument. The callable must return a tuple of
  // either arrays or None values.
  // TODO(phawkins): pass operand_shapes and result_shapes as
  // absl::Span<Shape const> when nanobind transition is complete.
  absl::StatusOr<std::pair<uint64_t, nanobind::object>>
  GetEmitPythonCallbackDescriptor(nanobind::callable callable,
                                  nanobind::object operand_shapes,
                                  nanobind::object result_shapes);
  // Deprecated; please switch to emitting a `CustomCallOp` directly.
  absl::StatusOr<XlaOp> EmitPythonCallbackFromDescriptor(
      XlaBuilder& builder, uint64_t descriptor,
      absl::Span<XlaOp const> operands, absl::Span<Shape const> result_shapes,
      std::optional<std::vector<Shape>> operand_layouts, bool has_side_effect);
  // Deprecated; please switch to using `GetEmitPythonCallbackDescriptor`
  // and then emitting a `CustomCall` op instead.
  absl::StatusOr<std::pair<XlaOp, nanobind::object>> EmitPythonCallback(
      nanobind::callable callable, XlaBuilder& builder,
      absl::Span<XlaOp const> operands, absl::Span<Shape const> result_shapes,
      std::optional<std::vector<Shape>> operand_layouts, bool has_side_effect);

  // `MakePythonCallbackUsingHostSendAndRecv` takes in an input Python callable
  // that takes in arguments of shapes `operand_shapes` and returns results of
  // shapes `result_shapes`. The arguments correspond to Send ops in the HLO
  // program through `send_channel_ids` and the results correspond to Recv ops
  // through `recv_channel_ids`. It returns the host callback as an opaque
  // object whose reference will keep the Python callback alive. The host
  // callback can be passed to `PyClient::Compile` or
  // `PyClient::DeserializeExecutable`. The corresponding Send/Recv ops in the
  // XLA computation can trigger the execution of this host callback.
  // `serializer` is a function that takes `callable` as an argument and returns
  // a serialized callable as a string.
  //
  // The callable receives as arguments NumPy arrays for arguments with array
  // types, and None for Token argument. The callable must return a tuple of
  // either arrays or None values.
  absl::StatusOr<nanobind::object> MakePythonCallbackUsingHostSendAndRecv(
      nanobind::callable callable, absl::Span<Shape const> operand_shapes,
      absl::Span<Shape const> result_shapes,
      absl::Span<uint16_t const> send_channel_ids,
      absl::Span<uint16_t const> recv_channel_ids,
      nanobind::callable serializer);

  std::vector<nanobind::object> LiveArrays() const;

  static void RegisterPythonTypes(nanobind::module_& m);

 protected:
  static void Initialize(nb_class_ptr<PyClient> client);

 private:
  friend class PyLoadedExecutable;
  friend class PyArray;
  friend struct PyArray_Storage;

  static int tp_traverse(PyObject* self, visitproc visit, void* arg);
  static int tp_clear(PyObject* self);
  static PyType_Slot slots_[];

  std::shared_ptr<ifrt::Client> ifrt_client_;
  absl::flat_hash_map<std::string, xla::ifrt::Client::ClientAttribute>
      client_attributes_;
  // Pointers to intrusive doubly-linked lists of arrays and executables, used
  // to iterate over all known objects when heap profiling. The list structure
  // is protected by the GIL.

  PyLoadedExecutable* executables_ = nullptr;
  PyArray_Storage* arrays_ = nullptr;

  absl::flat_hash_map<ifrt::Device*, nb_class_ptr<PyDevice>> devices_;
  absl::flat_hash_map<PjRtMemorySpace*, nb_class_ptr<PyMemorySpace>>
      memory_spaces_;
};

}  // namespace xla

#endif  // XLA_PYTHON_PY_CLIENT_H_
