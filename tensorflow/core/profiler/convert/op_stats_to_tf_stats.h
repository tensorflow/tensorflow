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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_TF_STATS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_TF_STATS_H_

#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "plugin/tensorboard_plugin_profile/protobuf/op_stats.pb.h"  // from @org_xprof
#include "plugin/tensorboard_plugin_profile/protobuf/tf_stats.pb.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {

TfStatsDatabase ConvertOpStatsToTfStats(const OpStats& op_stats);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_TF_STATS_H_
