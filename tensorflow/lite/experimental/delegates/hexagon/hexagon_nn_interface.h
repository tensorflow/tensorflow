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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_HEXAGON_NN_INTERFACE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_HEXAGON_NN_INTERFACE_H_

#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/hexagon_nn.h"

using hexagon_nn_config_fn = decltype(hexagon_nn_config);
using hexagon_nn_init_fn = decltype(hexagon_nn_init);

using hexagon_nn_set_powersave_level_fn =
    decltype(hexagon_nn_set_powersave_level);

using hexagon_nn_set_debug_level_fn = decltype(hexagon_nn_set_debug_level);

using hexagon_nn_prepare_fn = decltype(hexagon_nn_prepare);

using hexagon_nn_append_node_fn = decltype(hexagon_nn_append_node);

using hexagon_nn_append_const_node_fn = decltype(hexagon_nn_append_const_node);

using hexagon_nn_execute_fn = decltype(hexagon_nn_execute);

using hexagon_nn_execute_new_fn = decltype(hexagon_nn_execute_new);

using hexagon_nn_teardown_fn = decltype(hexagon_nn_teardown);

using hexagon_nn_snpprint_fn = decltype(hexagon_nn_snpprint);

using hexagon_nn_getlog_fn = decltype(hexagon_nn_getlog);

using hexagon_nn_get_perfinfo_fn = decltype(hexagon_nn_get_perfinfo);

using hexagon_nn_reset_perfinfo_fn = decltype(hexagon_nn_reset_perfinfo);

using hexagon_nn_op_id_to_name_fn = decltype(hexagon_nn_op_id_to_name);

using hexagon_nn_global_teardown_fn = decltype(hexagon_nn_global_teardown);

using hexagon_nn_global_init_fn = decltype(hexagon_nn_global_init);

using hexagon_nn_is_device_supported_fn =
    decltype(hexagon_nn_is_device_supported);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_HEXAGON_NN_INTERFACE_H_
