/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Support/MlirOptMain.h"
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback.h"
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_async.h"
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_sync.h"
#include "tfrt/init_tfrt_dialects.h"  // from @tf_runtime

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  tfrt::RegisterTFRTDialects(registry);
  registry.insert<tfrt::fallback_async::FallbackAsyncDialect>();
  registry.insert<tfrt::fallback::FallbackDialect>();
  registry.insert<tfrt::fallback_sync::FallbackSyncDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "TFRT pass driver", registry));
}
