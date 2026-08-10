// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
// RUN: not tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=128,10 -tf-xla-emit-use-tuple-args -tf-xla-emit-return-tuple 2>&1 | FileCheck %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main(%arg0: tensor<128x8xf32> {mhlo.sharding = "bad_sharding"}) {
    func.return
  }
}

// CHECK: failed to parse sharding 'bad_sharding'
