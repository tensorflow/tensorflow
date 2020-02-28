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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_HEXAGON_NN_HEXAGON_NN_INIT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_HEXAGON_NN_HEXAGON_NN_INIT_H_

#ifdef __cplusplus
extern "C" {
#endif
void hexagon_nn_global_teardown(void);
void hexagon_nn_global_init(void);
bool hexagon_nn_is_device_supported();
#ifdef __cplusplus
}
#endif
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_HEXAGON_NN_HEXAGON_NN_INIT_H_
