#!/bin/bash

_DL_FLD=${TF_TESTING_FL:-'/'}
cd "${_DL_FLD}" || exit 2


clone_benchmark() {
    local benchmarks_repo='https://github.com/tensorflow/benchmarks'
    local tf_ver
    tf_ver=$(grep -oP "_VERSION = '\K[0-9]+\.[0-9]+" \
        "${_DL_FLD}/tensorflow/tensorflow/tools/pip_package/setup.py")
    if [ -z "${tf_ver}" ]; then
        echo "TF version cannot be found..."
        exit 1
    else
        echo "tf_ver=${tf_ver}"
    fi
    # For TF version higher than 2.12, use the new AMD benchmarks repo
    if (( ${tf_ver//./} > 212 ));then
        benchmarks_repo='https://github.com/ROCmSoftwarePlatform/benchmarks'
        benchmark_branch='-b tf2.13-compatible'
    fi
    git clone ${benchmark_branch} ${benchmarks_repo}
}

git clone https://github.com/tensorflow/models.git
git clone https://github.com/tensorflow/examples.git
git clone https://github.com/tensorflow/autograph.git
clone_benchmark
