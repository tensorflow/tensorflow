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

#include "tensorflow/compiler/xla/python/outfeed_receiver_py.h"

#include <cstdint>
#include <memory>

#include "absl/algorithm/container.h"
#include "absl/synchronization/mutex.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/functional.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/outfeed_receiver.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/types.h"

namespace xla {

namespace py = pybind11;

namespace {

// A wrapper for OutfeedReceiver for use from Python, useful for ensuring
// that the GIL is released before destroying the OutfeedReceiver.
class OutfeedReceiverForPython {
 public:
  // A callback to Python takes: consumer id, received literal.
  using CallbackToPython =
      std::function<void(ClientAndPtr<PjRtDevice>, uint32_t, pybind11::object)>;

  OutfeedReceiverForPython(CallbackToPython callback_python,
                           std::vector<std::shared_ptr<PyClient>> clients,
                           ssize_t max_callback_queue_size_bytes)
      : callback_python_(std::move(callback_python)),
        clients_(std::move(clients)) {
    OutfeedReceiver::Callback callback =
        [this](PjRtDevice* device, uint32_t consumer_id,
               std::shared_ptr<Literal> literal) {
          this->Callback(device, consumer_id, std::move(literal));
        };
    std::vector<PjRtClient*> client_ptrs(clients_.size());
    absl::c_transform(clients_, client_ptrs.begin(),
                      [](const std::shared_ptr<PyClient>& client) {
                        return client->pjrt_client();
                      });
    outfeed_receiver_ = std::make_unique<OutfeedReceiver>(
        callback, client_ptrs, max_callback_queue_size_bytes);
  }
  OutfeedReceiverForPython(const OutfeedReceiverForPython&) = delete;
  OutfeedReceiverForPython& operator=(const OutfeedReceiverForPython&) = delete;

  ~OutfeedReceiverForPython() {
    // This destructor is called from the Python GC. Release it for the duration
    // of the destruction, including the destruction of the OutfeedReceiver,
    // when we may actually have to wait for threads to end. During this time
    // we do not callback to Python (sometimes we get an exception
    // "std::runtime_error: scoped_acquire::dec_ref(): thread state must
    // be current!"").
    {
      absl::MutexLock lock(&mu_);
      outfeed_receiver_shutting_down_ = true;
    }
    py::gil_scoped_release gil_release;
    outfeed_receiver_ = nullptr;  // Shutdown the outfeed receiver.
  }

  void Start() { outfeed_receiver_->Start(); }

  StatusOr<XlaOp> AddOutfeed(XlaBuilder* builder, XlaOp token,
                             uint32_t consumer_id, std::vector<XlaOp> arrays,
                             uint32_t device_idx) {
    return outfeed_receiver_->AddOutfeedToBuilder(builder, token, consumer_id,
                                                  arrays, device_idx);
  }

  void Callback(PjRtDevice* device, uint32_t consumer_id,
                std::shared_ptr<Literal> literal) {
    {
      absl::MutexLock lock(&mu_);
      if (outfeed_receiver_shutting_down_) {
        VLOG(2) << "Ignoring unsafe callback to Python during shutdown";
        return;
      }
    }
    // We expect the number of clients to be small, so an O(n) search is fine.
    auto it = absl::c_find_if(
        clients_, [device](const std::shared_ptr<PyClient>& client) {
          return client->pjrt_client() == device->client();
        });
    CHECK(it != clients_.end());
    py::gil_scoped_acquire gil_acquire;  // Need GIL also for LiteralToPython
    py::object literal_python = LiteralToPython(std::move(literal)).value();
    // The callback_ should handle all exceptions in user-code. If we get
    // an exception here, it is a bug in the callback and we should stop.
    callback_python_(WrapWithClient<PjRtDevice>(*it, device), consumer_id,
                     std::move(literal_python));
  }

 private:
  CallbackToPython callback_python_;
  absl::Mutex mu_;
  bool outfeed_receiver_shutting_down_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<std::shared_ptr<PyClient>> clients_;
  std::unique_ptr<OutfeedReceiver> outfeed_receiver_;
};

}  // namespace

void BuildOutfeedReceiverSubmodule(py::module* m) {
  py::module outfeed_receiver =
      m->def_submodule("outfeed_receiver", "Outfeed receiver");
  outfeed_receiver.def(
      "start",
      [](OutfeedReceiverForPython::CallbackToPython callback_to_python,
         std::vector<std::shared_ptr<PyClient>> clients,
         ssize_t max_callback_queue_size_bytes)
          -> std::unique_ptr<OutfeedReceiverForPython> {
        auto server = std::make_unique<OutfeedReceiverForPython>(
            callback_to_python, clients, max_callback_queue_size_bytes);
        server->Start();
        return server;
      },
      py::arg("callback_to_python"), py::arg("backends"),
      py::arg("max_queue_size_bytes") = 256 * 1024 * 1024,
      R"(Starts a multithreaded outfeed receiver.

      There is one thread for each of the specified devices. When Python
      drops the last reference to the returned object, the receiver is shut
      down. The destructor will block until all data is received from
      devices.

      Args:
        * callback_to_python: a Python callback to call, with <consumer_id>
          and the data received.
        * backends: the list of backends to listen on.
        * max_queue_size_bytes: an optional integer to bound the maximum size
            of arrays in the callback queue. When this limit is reached the
            device listener pauses.
      )",
      py::call_guard<py::gil_scoped_release>());

  py::class_<OutfeedReceiverForPython> outfeed_receiver_class(
      outfeed_receiver, "OutfeedReceiverForPython");

  outfeed_receiver_class.def(
      "add_outfeed", &OutfeedReceiverForPython::AddOutfeed, py::arg("builder"),
      py::arg("token"), py::arg("consumer_id"), py::arg("arrays"),
      py::arg("device_idx"),
      R"(Adds an outfeed into the given computation builder.

      Has the side-effect of registering the sent shape along with the consumer
      ID. Returns error if the outfeed shape is not compatible with previously
      used shape for the same consumer ID.)",
      py::call_guard<py::gil_scoped_release>());
}

}  // namespace xla
