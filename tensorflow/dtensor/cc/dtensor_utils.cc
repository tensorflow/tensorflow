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

#include "tensorflow/dtensor/cc/dtensor_utils.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/tsl/util/env_var.h"

namespace tensorflow {
namespace dtensor {

// LINT.IfChange
int ClientId() {
  char* client_id_str = std::getenv("DTENSOR_CLIENT_ID");
  if (client_id_str == nullptr) return 0;
  int client_id;
  if (absl::SimpleAtoi(client_id_str, &client_id)) return client_id;
  LOG(WARNING) << "Invalid DTENSOR_CLIENT_ID, using the default value 0.";
  return 0;
}
// LINT.ThenChange(//tensorflow/dtensor/python/dtensor_device.py)

// LINT.IfChange
int NumClients() {
  char* num_clients_str = std::getenv("DTENSOR_NUM_CLIENTS");
  if (num_clients_str == nullptr) return 1;
  int num_clients;
  if (absl::SimpleAtoi(num_clients_str, &num_clients)) return num_clients;
  LOG(WARNING) << "Invalid DTENSOR_NUM_CLIENTS, using the default value 1.";
  return 1;
}
// LINT.ThenChange(//tensorflow/dtensor/python/dtensor_device.py)

bool LogOnAllTasks() {
  char* dtensor_log_on_all_tasks_str = std::getenv("DTENSOR_LOG_ON_ALL_TASKS");
  if (dtensor_log_on_all_tasks_str == nullptr) return false;
  return true;
}

bool LogOpByOp(absl::string_view op_name) {
  char* op_list_str = std::getenv("DTENSOR_LOG_OP_BY_OP");
  if (op_list_str == nullptr) return false;
  if (!strcmp(op_list_str, "*")) return true;
  std::vector<absl::string_view> op_list = absl::StrSplit(op_list_str, ',');
  if (std::find(op_list.begin(), op_list.end(), op_name) != op_list.end()) {
    return true;
  }
  return false;
}

int LayoutPropagationMaxSteps() {
  char* dtensor_layout_propagation_max_steps_str =
      std::getenv("DTENSOR_LAYOUT_PROPAGATION_MAX_STEPS");
  if (dtensor_layout_propagation_max_steps_str == nullptr) return 500;
  int dtensor_layout_propagation_max_steps;
  if (absl::SimpleAtoi(dtensor_layout_propagation_max_steps_str,
                       &dtensor_layout_propagation_max_steps))
    return dtensor_layout_propagation_max_steps;
  LOG(WARNING) << "Invalid DTENSOR_LAYOUT_PROPAGATION_MAX_STEPS, using "
                  "the default value 500.";
  return 500;
}

bool EnableMixedPrecisionReduce() {
  char* dtensor_enable_mixed_precision_reduce_str =
      std::getenv("DTENSOR_ENABLE_MIXED_PRECISION_REDUCE");
  if (dtensor_enable_mixed_precision_reduce_str == nullptr) return false;
  return true;
}

bool DoNotFuseReduceScatter() {
  char* dtensor_do_not_fuse_reduce_scatter_str =
      std::getenv("DTENSOR_DO_NOT_FUSE_REDUCE_SCATTER");
  if (dtensor_do_not_fuse_reduce_scatter_str == nullptr) return false;
  return true;
}

int ReduceInBfloat16MaxGroupSize() {
  char* dtensor_reduce_in_bfloat16_max_group_size_str =
      std::getenv("DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE");
  if (dtensor_reduce_in_bfloat16_max_group_size_str == nullptr) return 8;
  int dtensor_reduce_in_bfloat16_max_group_size;
  if (absl::SimpleAtoi(dtensor_reduce_in_bfloat16_max_group_size_str,
                       &dtensor_reduce_in_bfloat16_max_group_size))
    return dtensor_reduce_in_bfloat16_max_group_size;
  LOG(WARNING) << "Invalid DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE, using "
                  "the default value 8.";
  return 8;
}

bool LowerCollectiveGatherToCollectiveGatherV2() {
  // We lower DTensorGather to CollectiveReduceV2 ops instead of
  // CollectiveGatherV2, since we do not observe a performance gain with Gather
  // lowering and ReduceV2 is agnostic of the rank order.
  //
  // If LOWER_DTENSOR_GATHER_TO_COLLECTIVE_GATHER_V2 environment variable is set
  // to '1', it is reduced to collective
  char* use_collective_gather =
      std::getenv("LOWER_DTENSOR_GATHER_TO_COLLECTIVE_GATHER_V2");
  if (use_collective_gather == nullptr) return false;
  return true;
}

bool EnableReplicatedSpmdAsDefault(const std::string& op_name) {
  // These environment variables enroll MLIR ops of the given name for default
  // replicated SPMD expansion. No expanders are registered for these Ops,
  // and without enrolling to the default replicated behavior, SPMD expansion
  // raises an error for these Op.
  //
  // For example, to enroll tf.Mod, set
  //   DTENSOR_ENABLE_REPLICATED_SPMD_AS_DEFAULT_TF.MOD = 1
  std::string env_name = "DTENSOR_ENABLE_REPLICATED_SPMD_AS_DEFAULT_" +
                         absl::AsciiStrToUpper(op_name);
  char* dtensor_enable_replicated_spmd_as_default =
      std::getenv(env_name.c_str());
  return dtensor_enable_replicated_spmd_as_default != nullptr;
}

bool EnableAllToAllForRelayout() {
  // Whether to use all-to-all collective for relayout when possible.
  static bool is_enabled = [] {
    bool ret = true;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("DTENSOR_USE_ALL_TO_ALL_RELAYOUT",
                                        /*default_val=*/true, &ret));
    return ret;
  }();
  return is_enabled;
}

}  // namespace dtensor
}  // namespace tensorflow
