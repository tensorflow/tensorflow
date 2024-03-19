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

#include "xla/python/outfeed_receiver_py.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/function.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/optional.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_builder.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/outfeed_receiver.h"
#include "xla/python/py_client.h"
#include "xla/python/types.h"
#include "tsl/platform/logging.h"

namespace xla {

namespace nb = nanobind;

namespace {

// A wrapper for OutfeedReceiver for use from Python, useful for ensuring
// that the GIL is released before destroying the OutfeedReceiver.
class OutfeedReceiverForPython {
 public:
  // A callback to Python takes: consumer id, received literal.
  using CallbackToPython =
      std::function<void(nb_class_ptr<PyDevice>, uint32_t, nb::object)>;

  OutfeedReceiverForPython(
      CallbackToPython callback_python,
      std::vector<nb_class_ptr<PyClient>> clients,
      ssize_t max_callback_queue_size_bytes,
      const std::optional<ExecutableBuildOptions>& executable_build_options)
      : callback_python_(std::move(callback_python)),
        clients_(std::move(clients)) {
    OutfeedReceiver::Callback callback =
        [this](PjRtDevice* device, uint32_t consumer_id,
               std::shared_ptr<Literal> literal) {
          this->Callback(device, consumer_id, std::move(literal));
        };
    std::vector<PjRtClient*> client_ptrs(clients_.size());
    absl::c_transform(clients_, client_ptrs.begin(),
                      [](const nb_class_ptr<PyClient>& client) {
                        return client->pjrt_client();
                      });
    outfeed_receiver_ = std::make_unique<OutfeedReceiver>(
        callback, client_ptrs, max_callback_queue_size_bytes,
        executable_build_options);
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
    nb::gil_scoped_release gil_release;
    outfeed_receiver_ = nullptr;  // Shutdown the outfeed receiver.
  }

  void Start() { outfeed_receiver_->Start(); }

  absl::StatusOr<XlaOp> AddOutfeed(XlaBuilder* builder, XlaOp token,
                                   uint32_t consumer_id,
                                   std::vector<XlaOp> arrays,
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
        clients_, [device](const nb_class_ptr<PyClient>& client) {
          return client->pjrt_client() == device->client();
        });
    CHECK(it != clients_.end());
    PyClient* client = it->get();
    nb::gil_scoped_acquire gil_acquire;  // Need GIL also for LiteralToPython
    nb::object literal_python = LiteralToPython(std::move(literal)).value();
    // The callback_ should handle all exceptions in user-code. If we get
    // an exception here, it is a bug in the callback and we should stop.
    callback_python_(client->GetPyDevice(device), consumer_id,
                     std::move(literal_python));
  }

 private:
  CallbackToPython callback_python_;
  absl::Mutex mu_;
  bool outfeed_receiver_shutting_down_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<nb_class_ptr<PyClient>> clients_;
  std::unique_ptr<OutfeedReceiver> outfeed_receiver_;
};

}  // namespace

void BuildOutfeedReceiverSubmodule(nb::module_& m) {
  nb::module_ outfeed_receiver =
      m.def_submodule("outfeed_receiver", "Outfeed receiver");
  outfeed_receiver.def(
      "start",
      [](OutfeedReceiverForPython::CallbackToPython callback_to_python,
         nb::sequence clients, ssize_t max_callback_queue_size_bytes,
         std::optional<ExecutableBuildOptions> executable_build_options)
          -> std::unique_ptr<OutfeedReceiverForPython> {
        auto server = std::make_unique<OutfeedReceiverForPython>(
            std::move(callback_to_python),
            SequenceToVector<nb_class_ptr<PyClient>>(clients),
            max_callback_queue_size_bytes, executable_build_options);
        nb::gil_scoped_release gil_release;
        server->Start();
        return server;
      },
      nb::arg("callback_to_python"), nb::arg("backends"),
      nb::arg("max_queue_size_bytes") = 256 * 1024 * 1024,
      nb::arg("executable_build_options").none() = nb::none(),
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
      )");

  nb::class_<OutfeedReceiverForPython> outfeed_receiver_class(
      outfeed_receiver, "OutfeedReceiverForPython");

  outfeed_receiver_class.def(
      "add_outfeed",
      xla::ValueOrThrowWrapper(&OutfeedReceiverForPython::AddOutfeed),
      nb::arg("builder"), nb::arg("token"), nb::arg("consumer_id"),
      nb::arg("arrays"), nb::arg("device_idx"),
      R"(Adds an outfeed into the given computation builder.

      Has the side-effect of registering the sent shape along with the consumer
      ID. Returns error if the outfeed shape is not compatible with previously
      used shape for the same consumer ID.)",
      nb::call_guard<nb::gil_scoped_release>());
}

}  // namespace xla
