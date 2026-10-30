/* Copyright 2026 The OpenXLA Authors.

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

#include <atomic>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/record_ffi.h"

namespace nb = nanobind;

namespace xla {
namespace {

std::atomic<bool> execute_called{false};
std::atomic<bool> record_called{false};

ffi::Error Execute(ffi::ResultBufferR1<ffi::F32>) {
  execute_called = true;
  return ffi::Error::Success();
}

ffi::Error Record(ffi::RecordContext context, ffi::ResultBufferR1<ffi::F32>) {
  record_called = true;
  return context.RequestStreamCapture();
}

XLA_FFI_DEFINE_HANDLER(ExecuteHandler, Execute,
                       ffi::Ffi::Bind().Ret<ffi::BufferR1<ffi::F32>>(),
                       {ffi::Traits::kCmdBufferCompatible});
XLA_FFI_DEFINE_HANDLER(RecordHandler, Record,
                       ffi::Ffi::BindRecord()
                           .Ctx<ffi::Extension<ffi::RecordExtension>>()
                           .Ret<ffi::BufferR1<ffi::F32>>());

}  // namespace

NB_MODULE(python_hlo_runner_ffi_test_lib, m) {
  m.def("handlers", [] {
    nb::dict handlers;
    handlers["execute"] = nb::capsule(reinterpret_cast<void*>(ExecuteHandler));
    handlers["record"] = nb::capsule(reinterpret_cast<void*>(RecordHandler));
    return handlers;
  });
  m.def("reset", [] {
    execute_called = false;
    record_called = false;
  });
  m.def("calls", [] {
    return nb::make_tuple(execute_called.load(), record_called.load());
  });
}

}  // namespace xla
