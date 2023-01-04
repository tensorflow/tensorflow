/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_ENTRYPOINT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_ENTRYPOINT_H_

// This is a helper function used by the runner_main.c to run validation test
// through commandline flags.
// - argv[1]: Not used. helper binary name.
// - argv[2]: Not used. entrypoint function name.
// - argv[3]: model path. This path will be used by ModelLoader to create model.
// - argv[4]: storage path for reading Minibenchmark settings and writing
// Minibenchmark result to.
// - argv[5]: Not used. data directory path.
// - argv[6]: Optional. NNAPI SL path.
// This function ensures thread-safety by creating a file lock of
// storage_path.child_lock, hence there will be at most one validation test
// running for a unique storage_path. The validation error and output is written
// to storage_path too.
extern "C" int Java_org_tensorflow_lite_acceleration_validation_entrypoint(
    int argc, char** argv);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_ENTRYPOINT_H_
