/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_XLA_OPS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_XLA_OPS_H_

#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/macros.h"

// RAII helper to set the frontend attribute for the target chip to the SC.
// Automatically restores the frontend attributes on exit.
class UseSparseCoreFrontendAttributes {
 public:
  explicit UseSparseCoreFrontendAttributes(xla::XlaBuilder* builder)
      : builder_(builder),
        original_frontend_attributes_(builder->frontend_attributes()) {
    xla::FrontendAttributes sc_attributes = original_frontend_attributes_;
    (*sc_attributes.mutable_map())["_xla_compute_type"] = "sparse";
    builder_->SetFrontendAttributes(sc_attributes);
  }

  ~UseSparseCoreFrontendAttributes() {
    builder_->SetFrontendAttributes(original_frontend_attributes_);
  }

 private:
  xla::XlaBuilder* builder_;
  const xla::FrontendAttributes original_frontend_attributes_;

  UseSparseCoreFrontendAttributes(const UseSparseCoreFrontendAttributes&) =
      delete;
  void operator=(const UseSparseCoreFrontendAttributes&) = delete;
};

#endif  // TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_XLA_OPS_H_
