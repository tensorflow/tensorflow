/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_TF_DATA_MEMORY_LOGGER_H_
#define TENSORFLOW_CORE_DATA_TF_DATA_MEMORY_LOGGER_H_

namespace tensorflow {
namespace data {

// Starts the iterator memory logger if it is not already started. The logger is
// only active at VLOG level 4.
void EnsureIteratorMemoryLoggerStarted();
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_TF_DATA_MEMORY_LOGGER_H_
