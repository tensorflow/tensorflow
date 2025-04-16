# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# TODO: Implement this script to leverage integration tests in OSS.

# This script must handle the following flags:

# DEFINE_string --required bin "" "The binary to execute on the device."
# DEFINE_array data --type=string "" "The data files to install on the device."
# DEFINE_bool do_exec false "Whether to execute the target on the device."
# DEFINE_array exec_args --type=string "" "The arguments to pass to the executable on device."
# DEFINE_array exec_env_vars --type=string "" "The environment variables to set for the executable on device."
# DEFINE_string device_rlocation_root "/data/local/tmp/runfiles" "The root directory for device relative locations."

# This script must push the bin file and all the data files to the device under
# the device_rlocation_root directory. If do_exec is true, it must execute the
# binary on the device with the given exec_args and exec_env_vars.