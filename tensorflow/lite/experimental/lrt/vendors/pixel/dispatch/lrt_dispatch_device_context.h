// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_PIXEL_DISPATCH_LRT_DISPATCH_DEVICE_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_PIXEL_DISPATCH_LRT_DISPATCH_DEVICE_CONTEXT_H_

#include <memory>
#include <set>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/vendors/pixel/dispatch/southbound.h"

class LrtDispatchDeviceContextT {
 public:
  ~LrtDispatchDeviceContextT();

  static absl::StatusOr<std::unique_ptr<LrtDispatchDeviceContextT>> Create(
      const lrt::pixel::Southbound& southbound);

  ThrContext* thr_context() { return thr_context_; }
  void add_graph(ThrGraph* graph) { thr_graphs_.insert(graph); }
  void remove_graph(ThrGraph* graph) { thr_graphs_.erase(graph); }

 private:
  explicit LrtDispatchDeviceContextT(const lrt::pixel::Southbound& southbound)
      : southbound_(southbound) {}

  const lrt::pixel::Southbound& southbound_;
  ThrContext* thr_context_ = nullptr;
  std::set<ThrGraph*> thr_graphs_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_PIXEL_DISPATCH_LRT_DISPATCH_DEVICE_CONTEXT_H_
