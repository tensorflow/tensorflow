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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/program.h"
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
  const std::shared_ptr<ifrt::Client>& shared_ptr_ifrt_client() const {
    return ifrt_client_;
  }

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

  absl::string_view platform_name() const {
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
  absl::string_view raw_platform_name() const {
    // TODO(parkers): Once platform_name() is the same, remove this.
    return ifrt_client_->platform_name();
  }
  absl::string_view platform_version() const {
    return ifrt_client_->platform_version();
  }
  absl::string_view runtime_type() const {
    return ifrt_client_->runtime_type();
  }

  // Returns implementation-specific attributes about this client, e.g. the PJRT
  // C API version if applicable.
  const xla::ifrt::AttributeMap& Attributes() const {
    return client_attributes_;
  }

  int addressable_device_count() const {
    return ifrt_client_->addressable_device_count();
  }
  int device_count() const { return ifrt_client_->device_count(); }
  int process_index() const { return ifrt_client_->process_index(); }

  std::vector<nb_class_ptr<PyDevice>> Devices();
  std::vector<nb_class_ptr<PyDevice>> LocalDevices();
  // Returns all devices in the client. Private API; only use this method for
  // implementing backend._get_all_devices().
  // TODO(hyeontaek): Remove this method once we have a unified API for
  // enumerating devices with different criteria.
  std::vector<nb_class_ptr<PyDevice>> GetAllDevices();
  absl::StatusOr<nb_class_ptr<PyDevice>> DeviceFromLocalHardwareId(
      int local_hardware_id);

  // Returns the PyDevice associated with the given ifrt::Device.
  nb_class_ptr<PyDevice> GetPyDevice(ifrt::Device* device);

  // Returns the PyMemorySpace associated with the given ifrt::Memory.
  nb_class_ptr<PyMemorySpace> GetPyMemorySpace(ifrt::Memory* memory_space);

  // Returns a vector of live PyArray objects. PyArray objects may share
  // PjRtBuffers, so there may be duplicates of the same underlying device
  // buffer.
  std::vector<nanobind::object> LiveBuffersOnDevice(ifrt::Device* device);

  nanobind::list LiveExecutables();

  // TODO(zhangqiaorjc): Remove when we have transparent defragmentation.
  absl::Status Defragment();

  static absl::StatusOr<nanobind::object> BufferFromPyval(
      nb_class_ptr<PyClient> client, nanobind::handle argument,
      ifrt::Device* device, bool force_copy,
      ifrt::Client::HostBufferSemantics host_buffer_semantics);

  static absl::StatusOr<nb_class_ptr<PyLoadedExecutable>> CompileIfrtProgram(
      nb_class_ptr<PyClient> client,
      std::unique_ptr<ifrt::Program> ifrt_program,
      std::unique_ptr<ifrt::CompileOptions> ifrt_options);

  static absl::StatusOr<nb_class_ptr<PyLoadedExecutable>> Compile(
      nb_class_ptr<PyClient> client, std::string mlir_module,
      CompileOptions options, std::vector<nanobind::capsule> host_callbacks);

  static absl::StatusOr<nb_class_ptr<PyLoadedExecutable>> Compile(
      nb_class_ptr<PyClient> client, std::string mlir_module,
      CompileOptions options, std::vector<nanobind::callable> host_callbacks);

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
  absl::StatusOr<std::pair<uint64_t, nanobind::object>>
  GetEmitPythonCallbackDescriptor(nanobind::callable callable,
                                  absl::Span<Shape const> operand_shapes,
                                  absl::Span<Shape const> result_shapes);

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

  std::vector<PyArray> LiveArrays() const;

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
  xla::ifrt::AttributeMap client_attributes_;
  // Pointers to intrusive doubly-linked lists of arrays and executables, used
  // to iterate over all known objects when heap profiling. The list structure
  // is protected by the GIL.

  nanobind::ft_mutex executables_mutex_;
  // List guarded by executables_mutex_.
  PyLoadedExecutable* executables_ = nullptr;

#ifdef NB_FREE_THREADING
  static constexpr size_t kNumArraysShards = 16;
#else
  static constexpr size_t kNumArraysShards = 1;
#endif
  struct ArraysShard {
    mutable nanobind::ft_mutex mutex;
    PyArray_Storage* arrays;
  };
  std::array<ArraysShard, kNumArraysShards> arrays_;

  absl::flat_hash_map<ifrt::Device*, nb_class_ptr<PyDevice>> devices_;
  absl::flat_hash_map<ifrt::Memory*, nb_class_ptr<PyMemorySpace>>
      memory_spaces_;
};

}  // namespace xla

#endif  // XLA_PYTHON_PY_CLIENT_H_
