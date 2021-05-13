# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# Needs to be run when compiled with -O0
file ../../../tools/make/gen/apollo3evb_cortex-m4/bin/preprocessor_1k_micro_test
target remote localhost:2331
load ../../../tools/make/gen/apollo3evb_cortex-m4/bin/preprocessor_1k_micro_test
monitor reset
break preprocessor.cc:211
commands
dump verilog value micro_windowed_input.txt fixed_input
dump verilog value micro_dft.txt fourier_values
dump verilog value micro_power.txt power_spectrum
dump verilog memory micro_power_avg.txt output output+42
c
end
break preprocessor_1k.cc:50
commands
print count
end
c
