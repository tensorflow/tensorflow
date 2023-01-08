#!/bin/bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# Benchmarks all downloaded models, parses, and summarizes results.

set -x
source onednn_benchmark_config.sh

export OUTDIR=~/onednn_benchmarks
mkdir -p ${OUTDIR}
bash run_models.sh 2>&1 | tee ${OUTDIR}/verbose.log
grep -v 'profiler_session\|xplane' ${OUTDIR}/verbose.log > ${OUTDIR}/run.log
grep "\+ ${BUILDER}-bin\|no stats:\|'BATCH=" ${OUTDIR}/run.log > ${OUTDIR}/to_parse.log
python parse_onednn_benchmarks.py ${OUTDIR}/to_parse.log | tee ${OUTDIR}/results.csv
