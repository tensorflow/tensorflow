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
// RUN: tfg-translate -mlir-to-graphdef %s | FileCheck %s

tfg.graph #tf_type.version<producer = 42, min_consumer = 21, bad_consumers = [1, 2, 5, 12]> {
}

// CHECK: producer: 42
// CHECK: min_consumer: 21
// CHECK: bad_consumers: 1
// CHECK: bad_consumers: 2
// CHECK: bad_consumers: 5
// CHECK: bad_consumers: 12
