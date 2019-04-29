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
#include "tensorflow/lite/kernels/internal/backends/backend_context.h"

// Supported backends
#include "tensorflow/lite/kernels/internal/backends/acl/backend.h"

namespace tflite {
namespace internal {
namespace backends {

KernelBackendContext::KernelBackendContext() : _backends() {
  _backends.emplace("acl", std::unique_ptr<IBackend>(new acl::ACLBackend()));
}

KernelBackendContext::~KernelBackendContext() {}

IBackend *KernelBackendContext::backend(std::string backend) {
  if (_backends.count(backend)) {
    return _backends[backend].get();
  }
  return nullptr;
}

std::unique_ptr<IBackendKernel> KernelBackendContext::backend_kernel(
    std::string backend, TfLiteBuiltinOperator op) {
  if (_backends.count(backend) && _backends[backend] != nullptr) {
    return _backends[backend]->create_backend_kernel(op);
  }
  return nullptr;
}

std::unique_ptr<IBackendKernel> KernelBackendContext::backend_kernel(
    TfLiteBuiltinOperator op) {
  std::unique_ptr<IBackendKernel> kernel = nullptr;

  auto backend_it = _backends.cbegin();
  while (backend_it != _backends.cend() && kernel == nullptr) {
    IBackend *backend = backend_it->second.get();
    if (backend != nullptr) {
      kernel = backend->create_backend_kernel(op);
    }
    ++backend_it;
  }

  return kernel;
}

void KernelBackendContext::set_max_num_threads(std::string backend,
                                               int max_num_threads) {
  if (_backends.count(backend) && _backends[backend] != nullptr) {
    _backends[backend]->set_max_num_threads(max_num_threads);
  }
}

void KernelBackendContext::set_max_num_threads_all(int max_num_threads) {
  for (auto &backend_pair : _backends) {
    IBackend *backend = backend_pair.second.get();
    if (backend != nullptr) {
      backend->set_max_num_threads(max_num_threads);
    }
  }
}

}  // namespace backends
}  // namespace internal
}  // namespace tflite
