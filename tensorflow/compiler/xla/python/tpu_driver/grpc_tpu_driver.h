#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_GRPC_TPU_DRIVER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_GRPC_TPU_DRIVER_H_

// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "grpcpp/grpcpp.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"

namespace tpu_driver {

xla::StatusOr<std::unique_ptr<TpuDriver>> CreateGrpcTpuDriver(
    const TpuDriverConfig& config,
    std::shared_ptr<grpc_impl::ChannelCredentials> credentials);

}  // namespace tpu_driver

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_GRPC_TPU_DRIVER_H_
