#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================

set -e
set -x

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
# If rocm-smi exists locally (it should) use it to find
# out how many GPUs we have to test with.
rocm-smi -i
STATUS=$?
if [ $STATUS -ne 0 ]; then TF_GPU_COUNT=1; else
   TF_GPU_COUNT=$(rocm-smi -i|grep 'Device ID' |grep 'GPU' |wc -l)
fi
TF_TESTS_PER_GPU=1
N_TEST_JOBS=$(expr ${TF_GPU_COUNT} \* ${TF_TESTS_PER_GPU})

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s)."
echo ""

# First positional argument (if any) specifies the ROCM_INSTALL_DIR
if [[ -n $1 ]]; then
    ROCM_INSTALL_DIR=$1
else
    if [[ -z "${ROCM_PATH}" ]]; then
        ROCM_INSTALL_DIR=/opt/rocm/
    else
        ROCM_INSTALL_DIR=$ROCM_PATH
    fi
fi

export PYTHON_BIN_PATH=`which python3`
PYTHON_VERSION=`python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
export TF_PYTHON_VERSION=$PYTHON_VERSION
export TF_NEED_ROCM=1
export ROCM_PATH=$ROCM_INSTALL_DIR

if [ ! -d /tf ];then
        # The bazelrc files in /usertools expect /tf to exist
        mkdir /tf
    fi

# vvv TODO (rocm) weekly-sync-20251021 excluded tests
EXCLUDED_TESTS=(
    # @local_xla//xla/backends/gpu/codegen/triton:dot_algorithms_legacy_test_amdgpu_any
    # @local_xla//xla/backends/gpu/codegen/triton:dot_algorithms_test_amdgpu_any
    TritonAndBlasSupportForDifferentTensorSizes/TritonAndBlasSupportForDifferentTensorSizes.IsDotAlgorithmSupportedByTriton/dot_*

    # @local_xla//xla/backends/gpu/codegen/triton:fusion_emitter_device_legacy_port_test_amdgpu_any
    CompareTest.SplitK

    # @local_xla//xla/backends/gpu/codegen/triton:fusion_emitter_device_test_amdgpu_any
    TritonEmitterTest.FusionWithOutputContainingMoreThanInt32MaxElementsExecutesCorrectly
    TritonEmitterTest.ConvertF16ToF8E5M2Exhaustive
    TritonEmitterTest.RocmWarpSizeIsSetCorrectly
    BasicDotAlgorithmEmitterTestSuite/BasicDotAlgorithmEmitterTest.BasicAlgorithmIsEmittedCorrectly/ALG_DOT_F16_F16_F16

    # @local_xla//xla/backends/gpu/codegen/triton:fusion_emitter_int4_device_test_amdgpu_any
    TritonTest.FuseSubchannelDequantizationWithTranspose

    # @local_xla//xla/backends/gpu/codegen/triton:fusion_emitter_parametrized_test_amdgpu_any
    TritonNormalizationTest.CanFuseAndEmitDiamondWithBF16Converts
    ElementwiseTestSuiteF16/UnaryElementwiseTest.ElementwiseUnaryOpExecutesCorrectly/f16_cosine
    ElementwiseTestSuiteF16/BinaryElementwiseTest.ElementwiseBinaryOpExecutesCorrectly/f16_atan2
    ElementwiseTestSuiteF16/BinaryElementwiseTest.ElementwiseFusionExecutesCorrectly/f16_atan2

    # @local_xla//xla/service/gpu/tests:command_buffer_test_amdgpu_any
    CommandBufferTests/CommandBufferTest.WhileLoop/*
    CommandBufferTests/CommandBufferTest.IndexConditional/*
    CommandBufferTests/CommandBufferTest.TrueFalseConditional/*

    # @local_xla//xla/backends/gpu/runtime:command_buffer_conversion_pass_test_amdgpu_any
    CommandBufferConversionPassTest.ConvertWhileThunk
    CommandBufferConversionPassTest.ConvertWhileThunkWithAsyncPair

    # @local_xla//xla/backends/gpu/runtime:topk_test_amdgpu_any
    TopKTests/TopKKernelTest.*

    # @local_xla//xla/pjrt/c:pjrt_c_api_gpu_test_amdgpu_any
    PjrtCAPIGpuExtensionTest.TritonCompile

    # @local_xla//xla/service/gpu:dot_algorithm_support_test_amdgpu_any
    DotTf32Tf32F32Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_tf32_tf32_f32_*
    DotTf32Tf32F32X3Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_tf32_tf32_f32_*

    # @local_xla//xla/service/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any_notfrt
    # @local_xla//xla/service/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any
    TritonFusionNumericsVerifierTest.CompilationSucceedsEvenIfKernelWillSpillRegisters
    TritonFusionNumericsVerifierTest.VerifyThatDisablingTritonIsFast

    # @local_xla//xla/service/gpu/tests:gpu_cub_sort_test_amdgpu_any
    CubSortKeysTest.CompareToReferenceNumpyOrderGt
    CubSortKeysTest.CompareToReferenceTotalOrderLt
    CubSort/CubSortKeysTest.*
    CubSort/CubSortPairsTest.*

    # @local_xla//xla/backends/gpu/runtime:cub_sort_thunk_test
    CubSortThunkTest.ProtoRoundTrip

    # @local_xla//xla/service/gpu/transforms:cublas_gemm_rewriter_test_amdgpu_any
    CublasLtGemmRewriteTest.MatrixBiasSwishActivation
    CublasLtGemmRewriteTest.VectorBiasReluActivationF16Padded
    CublasLtGemmRewriteTest.VectorBiasF16Padded
    CublasLtGemmRewriteTest.ReluActivationF16Padded
    CublasLtGemmRewriteTest.VectorBiasReluActivationBF16Padded
    CublasLtGemmRewriteTest.BF16VectorBiasPadded
    CublasLtGemmRewriteTest.ApproxGeluActivationBF16
    CublasLtGemmRewriteTest.ReluActivationBF16Padded
    CublasLtGemmRewriteTest.VectorBiasBF16Padded

    # @local_xla//xla/service/gpu:determinism_test_amdgpu_any
    DeterminismTest.Conv

    # @local_xla//xla/tests:sample_file_test_amdgpu_any
    # @local_xla//xla/tests:sample_file_test_amdgpu_any_notfrt
    SampleFileTest.Convolution

    # @local_xla//xla/tests:scatter_deterministic_expander_test_amdgpu_any
    # @local_xla//xla/tests:scatter_deterministic_expander_test_amdgpu_any_notfrt
    # @local_xla//xla/tests:scatter_test_amdgpu_any
    # @local_xla//xla/tests:scatter_test_amdgpu_any_notfrt
    ScatterTest.TensorFlowScatterV1_UpdateTwice

    # @local_xla//xla/tests:multioutput_fusion_test_amdgpu_any
    MultiOutputFusionTest.MultiOutputReduceFusionMajorWithExtraOutput
)

bazel --bazelrc=tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/rocm.bazelrc test \
    --config=sigbuild_local_cache \
    --config=rocm \
    --config=xla_cpp_filters \
    --test_output=errors \
    --local_test_jobs=${N_TEST_JOBS} \
    --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
    --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
    --test_env=MIOPEN_FIND_ENFORCE=5 \
    --test_env=MIOPEN_FIND_MODE=1 \
    --action_env="ROCM_PATH=$ROCM_PATH" \
    --action_env=XLA_FLAGS=--xla_gpu_force_compilation_parallelism=16 \
    --test_filter=-$(IFS=: ; echo "${EXCLUDED_TESTS[*]}") \
    -- @local_xla//xla/... \
    -@local_xla//xla/service/gpu/tests:sorting_test_amdgpu_any \
    -@local_xla//xla/service/gpu/tests:sorting.hlo.test_mi200 \
    -@local_xla//xla/backends/gpu/codegen/emitters/tests:reduce_row/mof_scalar_variadic.hlo.test \
    -@local_xla//xla/backends/gpu/codegen/emitters/tests:reduce_row/side_output_broadcast.hlo.test \
    -@local_xla//xla/tools/hlo_opt:tests/gpu_hlo_llvm.hlo.test
    # ^^^ TODO (rocm) weekly-sync-20251021 excluded test files
