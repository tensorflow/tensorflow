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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_CLIENT_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#ifdef JAX_ENABLE_IFRT
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"
#endif
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class PyBuffer;
class PyShardedBuffer;
class PyClient;
class PyLoadedExecutable;
class PyArray;
struct PyArray_Storage;

// Custom holder types.
//
// We must keep the PyClient object alive as long as any of the runtime
// objects are alive. Since we don't have a lot of control over Python
// destructor ordering, we keep the PyClient object as a std::shared_ptr<>,
// and ensure that each Python runtime object holds a reference to the
// PyClient. An alternative design would be to keep a single global
// singleton PyClient, although this seems less flexible, especially for
// writing tests.
//
// To maintain PyClient references, we define pybind11 holder classes that
// are custom smart pointers that also keep a reference to a PyClient.
// pybind11 has a `keep_alive` feature that has a similar goal, but it doesn't
// seem sufficiently flexible to describe ownership relationships in cases where
// the ownership doesn't pertain to a direct argument or return value of a
// function. Another alternative to the holder classes would be to create proxy
// objects that contain both a reference and a runtime class; holder classes
// seem less tedious to define.

// A pair of a PyClient reference and an unowned pointer to T.
template <typename T>
struct ClientAndPtr {
  ClientAndPtr() = default;
  // pybind11 requires that we define a constructor that takes a raw pointer,
  // but it should be unreachable.
  explicit ClientAndPtr(T*) {
    LOG(FATAL) << "ClientAndPtr should constructed via WrapWithClient.";
  }

  ClientAndPtr(const ClientAndPtr&) = default;
  ClientAndPtr(ClientAndPtr&&) = default;
  ClientAndPtr& operator=(const ClientAndPtr&) = default;
  ClientAndPtr& operator=(ClientAndPtr&&) = default;

  std::shared_ptr<PyClient> client;
  T* contents;

  T* get() const { return contents; }
  T* operator->() const { return contents; }
  T& operator*() const { return *contents; }
};

// By defining a templated helper function, we can use return type deduction
// and avoid specifying types at the caller.
template <typename T>
ClientAndPtr<T> WrapWithClient(std::shared_ptr<PyClient> client, T* contents) {
  ClientAndPtr<T> result;
  result.client = std::move(client);
  result.contents = contents;
  return result;
}

// Python wrapper around PjRtClient.
// We use a wrapper class to add Python-specific functionality.
class PyClient : public std::enable_shared_from_this<PyClient> {
 public:
#ifdef JAX_ENABLE_IFRT
  explicit PyClient(std::unique_ptr<ifrt::Client> ifrt_client);
  explicit PyClient(std::shared_ptr<ifrt::Client> ifrt_client);
#else
  explicit PyClient(std::unique_ptr<PjRtClient> pjrt_client);
  explicit PyClient(std::shared_ptr<PjRtClient> pjrt_client);
#endif
  virtual ~PyClient();

#ifdef JAX_ENABLE_IFRT
  ifrt::Client* ifrt_client() const { return ifrt_client_.get(); }

  // Short-term escape hatch to get PjRtClient from PyClient.
  // TODO(hyeontaek): Migrate all users of this method to be agnostic of PjRt.
  xla::PjRtClient* pjrt_client() const {
    auto* pjrt_client =
        llvm::dyn_cast_or_null<ifrt::PjRtClient>(ifrt_client_.get());
    if (pjrt_client == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return pjrt_client->pjrt_client();
  }
  std::shared_ptr<PjRtClient> shared_ptr_pjrt_client() {
    auto* pjrt_client =
        llvm::dyn_cast_or_null<ifrt::PjRtClient>(ifrt_client_.get());
    if (pjrt_client == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return pjrt_client->shared_ptr_pjrt_client();
  }
#else
  xla::PjRtClient* pjrt_client() const { return pjrt_client_.get(); }
  std::shared_ptr<PjRtClient> shared_ptr_pjrt_client() { return pjrt_client_; }
#endif

  // Legacy alises.
  std::shared_ptr<PjRtClient> shared_pjrt_client() {
    return shared_ptr_pjrt_client();
  }

  absl::string_view platform_name() const {
#ifdef JAX_ENABLE_IFRT
    return ifrt_client_->platform_name();
#else
    return pjrt_client_->platform_name();
#endif
  }
  absl::string_view platform_version() const {
#ifdef JAX_ENABLE_IFRT
    return ifrt_client_->platform_version();
#else
    return pjrt_client_->platform_version();
#endif
  }
  absl::string_view runtime_type() const {
#ifdef JAX_ENABLE_IFRT
    return ifrt_client_->runtime_type();
#else
    return PjRtRuntimeTypeString(pjrt_client_->runtime_type());
#endif
  }
  int addressable_device_count() const {
#ifdef JAX_ENABLE_IFRT
    return ifrt_client_->addressable_device_count();
#else
    return pjrt_client_->addressable_device_count();
#endif
  }
  int device_count() const {
#ifdef JAX_ENABLE_IFRT
    return ifrt_client_->device_count();
#else
    return pjrt_client_->device_count();
#endif
  }
  int process_index() const {
#ifdef JAX_ENABLE_IFRT
    return ifrt_client_->process_index();
#else
    return pjrt_client_->process_index();
#endif
  }

  std::vector<ClientAndPtr<PjRtDevice>> Devices();
  std::vector<ClientAndPtr<PjRtDevice>> LocalDevices();

  // Returns a vector of live PyBuffer objects. PyBuffer objects may share
  // PjRtBuffers, so there may be duplicates of the same underlying device
  // buffer.
  std::vector<pybind11::object> LiveBuffers();
  std::vector<pybind11::object> LiveBuffersOnDevice(PjRtDevice* device);

  // Returns a vector of live PyLoadedExecutable objects.
  // note: must return std::shared_ptr instead of raw ptrs
  // https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#std-shared-ptr
  std::vector<std::shared_ptr<PyLoadedExecutable>> LiveExecutables();

  // TODO(zhangqiaorjc): Remove when we have transparent defragmentation.
  Status Defragment();

  StatusOr<std::vector<std::vector<ClientAndPtr<PjRtDevice>>>>
  GetDefaultDeviceAssignment(int num_replicas, int num_partitions);

  // TODO(skye): delete after all callers can handle 2D output
  StatusOr<std::vector<ClientAndPtr<PjRtDevice>>> GetDefaultDeviceAssignment1D(
      int num_replicas);

  StatusOr<ChannelHandle> CreateChannelHandle() { return ChannelHandle(); }
  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() {
#ifdef JAX_ENABLE_IFRT
    return ifrt_client_->CreateDeviceToHostChannelHandle();
#else
    return pjrt_client_->CreateDeviceToHostChannelHandle();
#endif
  }
  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() {
#ifdef JAX_ENABLE_IFRT
    return ifrt_client_->CreateHostToDeviceChannelHandle();
#else
    return pjrt_client_->CreateHostToDeviceChannelHandle();
#endif
  }

  StatusOr<std::vector<std::pair<pybind11::bytes, pybind11::object>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device);

  StatusOr<pybind11::object> BufferFromPyval(
      pybind11::handle argument, PjRtDevice* device, bool force_copy,
#ifdef JAX_ENABLE_IFRT
      ifrt::Client::HostBufferSemantics host_buffer_semantics
#else
      PjRtClient::HostBufferSemantics host_buffer_semantics
#endif
  );

  StatusOr<std::shared_ptr<PyLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options,
      std::vector<pybind11::capsule> host_callbacks);
  StatusOr<std::shared_ptr<PyLoadedExecutable>> CompileMlir(
      std::string mlir_module, CompileOptions options,
      std::vector<pybind11::capsule> host_callbacks);

  StatusOr<pybind11::bytes> SerializeExecutable(
      const PyLoadedExecutable& executable) const;
  StatusOr<std::shared_ptr<PyLoadedExecutable>> DeserializeExecutable(
      const std::string& serialized, CompileOptions options,
      std::vector<pybind11::capsule> host_callbacks);

  // TODO(skyewm): remove when jax stop providing hlo_module
  StatusOr<std::shared_ptr<PyLoadedExecutable>> DeserializeExecutable(
      const std::string& serialized, std::shared_ptr<HloModule> hlo_module,
      CompileOptions options, std::vector<pybind11::capsule> host_callbacks) {
    return DeserializeExecutable(serialized, options,
                                 std::move(host_callbacks));
  }

  StatusOr<pybind11::bytes> HeapProfile();

  // `GetEmitPythonCallbackDescriptor` takes in an input Python callable that
  // takes in arguments of shapes `operand_shapes` and returns values of shapes
  // `result_shapes`. It returns a pair of a `uint64_t` descriptor and a Python
  // object whose reference will keep the Python callback alive. The descriptor
  // should be passed into a 'xla_cpu_python_callback' CustomCall as its first
  // argument. Typically the callback may be kept alive by attaching the
  // keep-alive object to the executable built from this computation.
  //
  // The callable receives as arguments NumPy arrays for arguments with array
  // types, and None for Token argument. The callable must return a tuple of
  // either arrays or None values.
  //
  // This is a method of PyClient since different platforms may implement this
  // functionality in different ways.
  StatusOr<std::pair<uint64_t, pybind11::object>>
  GetEmitPythonCallbackDescriptor(pybind11::function callable,
                                  absl::Span<Shape const> operand_shapes,
                                  absl::Span<Shape const> result_shapes);
  // Deprecated; please switch to emitting an MHLO `CustomCallOp` directly.
  StatusOr<XlaOp> EmitPythonCallbackFromDescriptor(
      XlaBuilder& builder, uint64_t descriptor,
      absl::Span<XlaOp const> operands, absl::Span<Shape const> result_shapes,
      std::optional<std::vector<Shape>> operand_layouts, bool has_side_effect);
  // Deprecated; please switch to using `GetEmitPythonCallbackDescriptor`
  // and then emitting a `CustomCall` op instead.
  StatusOr<std::pair<XlaOp, pybind11::object>> EmitPythonCallback(
      pybind11::function callable, XlaBuilder& builder,
      absl::Span<XlaOp const> operands, absl::Span<Shape const> result_shapes,
      std::optional<std::vector<Shape>> operand_layouts, bool has_side_effect);

  // `MakePythonCallbackUsingHostSendAndRecv` takes in an input Python callable
  // that takes in arguments of shapes `operand_shapes` and returns results of
  // shapes `result_shapes`. The arguments correspond to Send ops in the HLO
  // program through `send_channel_ids` and the results correspond to Recv ops
  // through `recv_channel_ids`. It returns the host callback as an opaque
  // object whose reference will keep the Python callback alive. The host
  // callback can be passed to PyLoadedExecutable::Execute() so that the
  // corresponding Send/Recv ops can trigger the execution of this host
  // callback.
  StatusOr<pybind11::object> MakePythonCallbackUsingHostSendAndRecv(
      pybind11::function callable, absl::Span<Shape const> operand_shapes,
      absl::Span<Shape const> result_shapes,
      absl::Span<uint16_t const> send_channel_ids,
      absl::Span<uint16_t const> recv_channel_ids);

  std::vector<pybind11::object> LiveArrays();

 private:
  friend class PyBuffer;
  friend class PyShardedBuffer;
  friend class PyLoadedExecutable;
  friend class PyArray;
  friend struct PyArray_Storage;

#ifdef JAX_ENABLE_IFRT
  std::shared_ptr<ifrt::Client> ifrt_client_;
#else
  std::shared_ptr<PjRtClient> pjrt_client_;
#endif

  // Pointers to intrusive doubly-linked lists of buffers and executables, used
  // to iterate over all known objects when heap profiling. The list structure
  // is protected by the GIL.

  // buffers_ is a per-device list, indexed by device->id().
  std::vector<PyBuffer*> buffers_;
  PyLoadedExecutable* executables_ = nullptr;
  PyArray_Storage* arrays_ = nullptr;
  PyShardedBuffer* sharded_buffers_ = nullptr;
};

}  // namespace xla

PYBIND11_DECLARE_HOLDER_TYPE(T, xla::ClientAndPtr<T>);

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_CLIENT_H_
