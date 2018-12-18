# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

file ../../../tools/make/gen/apollo3evb_cortex-m4/bin/pushbutton_cmsis_speech_test
target remote localhost:2331
load ../../../tools/make/gen/apollo3evb_cortex-m4/bin/pushbutton_cmsis_speech_test
monitor reset
break pushbutton_main.c:313
commands
dump verilog value captured_data.txt captured_data
c
end
c
