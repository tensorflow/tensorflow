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
    # @xla//xla/backends/gpu/codegen/triton:dot_algorithms_test_amdgpu_any
    TritonAndBlasSupportForDifferentTensorSizes/TritonAndBlasSupportForDifferentTensorSizes.IsDotAlgorithmSupportedByTriton/dot_*

    # @xla//xla/backends/gpu/codegen/triton:fusion_emitter_int4_device_test_amdgpu_any
    TritonTest.FuseSubchannelDequantizationWithTranspose

    # @xla//xla/backends/gpu/codegen/triton:fusion_emitter_parametrized_test_amdgpu_any
    TritonNormalizationTest.CanFuseAndEmitDiamondWithBF16Converts
    ElementwiseTestSuiteF16/UnaryElementwiseTest.ElementwiseUnaryOpExecutesCorrectly/f16_cosine
    ElementwiseTestSuiteF16/BinaryElementwiseTest.ElementwiseBinaryOpExecutesCorrectly/f16_atan2
    ElementwiseTestSuiteF16/BinaryElementwiseTest.ElementwiseFusionExecutesCorrectly/f16_atan2

    # @xla//xla/backends/gpu/runtime:command_buffer_conversion_pass_test_amdgpu_any
    CommandBufferConversionPassTest.ConvertWhileThunk
    CommandBufferConversionPassTest.ConvertWhileThunkWithAsyncPair

    # @xla//xla/service/gpu:dot_algorithm_support_test_amdgpu_any
    DotTf32Tf32F32Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_tf32_tf32_f32_*
    DotTf32Tf32F32X3Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_tf32_tf32_f32_*

    # @xla//xla/service/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any_notfrt
    # @xla//xla/service/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any
    TritonFusionNumericsVerifierTest.CompilationSucceedsEvenIfKernelWillSpillRegisters
    TritonFusionNumericsVerifierTest.VerifyThatDisablingTritonIsFast
    TritonFusionNumericsVerifierTestSuite/TritonFusionNumericsVerifierTest.VerifyMultipleNestedFusionNumerics/0
    TritonFusionNumericsVerifierTestSuite/TritonFusionNumericsVerifierTest.VerifyMultipleNestedFusionNumerics/1

    # @xla//xla/service/gpu/transforms:dot_dimension_sorter_test_amdgpu_any
    WithoutDotDimensionSorterTest.UnsortedDimsCreateTransposes # failing on mi250
    WithoutDotDimensionSorterTest.DimOrderCanBeChanged # failing on mi250

    # @xla//xla/service/gpu/transforms:cublas_gemm_rewriter_test_amdgpu_any
    CublasLtGemmRewriteTest.MatrixBiasSwishActivation
    CublasLtGemmRewriteTest.VectorBiasReluActivationF16Padded
    CublasLtGemmRewriteTest.VectorBiasF16Padded
    CublasLtGemmRewriteTest.ReluActivationF16Padded
    CublasLtGemmRewriteTest.VectorBiasReluActivationBF16Padded
    CublasLtGemmRewriteTest.BF16VectorBiasPadded
    CublasLtGemmRewriteTest.ApproxGeluActivationBF16
    CublasLtGemmRewriteTest.ReluActivationBF16Padded
    CublasLtGemmRewriteTest.VectorBiasBF16Padded

    # @xla//xla/tests:sample_file_test_amdgpu_any
    # @xla//xla/tests:sample_file_test_amdgpu_any_notfrt
    SampleFileTest.Convolution

    # @xla//xla/tests:scatter_deterministic_expander_test_amdgpu_any
    # @xla//xla/tests:scatter_deterministic_expander_test_amdgpu_any_notfrt
    # @xla//xla/tests:scatter_test_amdgpu_any
    # @xla//xla/tests:scatter_test_amdgpu_any_notfrt
    ScatterTest.TensorFlowScatterV1_UpdateTwice

    # vvv TODO (rocm) weekly-sync-20251224 excluded tests

    # @xla//xla/backends/gpu/codegen/triton:support_test
    BitcastOrReshapeTestSuite/BitcastOrReshapeTest.IsTritonSupportedBitcastOrReshape*
    BitcastOrReshapeTestSuite/BitcastOrReshapeTest.IsTritonSupported0DBitcastOrReshape*
    BitcastConvertSuite/BitcastConvertTest.BitcastConvertDisguisedAsBitcast*
    UnaryElementwiseTestSuite/UnaryElementwiseTest.IsTritonSupportedUnaryElementwise*
    ConvertTestSuite/ConvertTest.Convert*
    BinaryElementwiseTestSuite/BinaryElementwiseTest.IsTritonSupportedBinaryElementwise*
    TernaryElementwiseTestSuite/TernaryElementwiseTest.IsTritonSupportedTernaryElementwise*
    ReductionComputationTestSuite/ReductionComputationTest.DifferentBinaryOps*
    TransposeTestSuite/TransposeTest.LoadTranspose3D*
    SliceTestSuite/SliceTest.ContinuousSlice*
    BroadcastTestSuite/BroadcastTest.Broadcast*
    ParameterTestSuite/ParameterTest.Parameter*
    ConstantTestSuite/ConstantTest.ConstantEffectiveScalar*
    DotTestSuite/DotTypesTest.Dot*

    # @xla//xla/backends/gpu/profiler:kernel_name_tracer_test
    KernelNameTracerTest.Create
    KernelNameTracerTest.CaptureKernelNames
    KernelNameTracerTest.CaptureKernelNamesFromCommandBufferThunk

    # @xla//xla/service/gpu/tests:swap_conv_operands_test
    SwapConvOperandsTest.LargePadding
    SwapConvOperandsTest.SmallPadding
    SwapConvOperandsTest.DoesNotLower

    # @xla//xla/service/gpu/tests:gpu_triton_custom_call_test
    GpuIrEmitterUnnestedTest.CanNotEmitTritonCustomCallOnPreAmpereGpu

    # @xla//xla/tests:convolution_autotune_disabled_test
    Transposed2DConvHloTest/Transposed2DConvHloTest.Simple*
    ConvolveWithAndWithoutCanonicalization_Instantiation/ConvolveWithAndWithoutCanonicalization.Convolve2D_NoSpatialDims*
    ConvolutionHloTest.ConvolveBackwardInput
    ConvolutionHloTest.TestConv0D
    ConvolutionHloTest.TestConv2DF16
    ConvolutionHloTest.SwappedOperandConvolveWithStride
    ConvolutionHloTest.TestFusedConv2D
    ConvolutionHloTest.TestFusedConv3D
    ConvolutionHloTest.SwappedOperandConvolve
    ConvolutionHloTest.TestBooleanInput
    ConvolutionHloTest.SwappedOperandConvolve2
    ConvolutionTest.Convolve3D_1x4x2x3x3_2x2x2x3x3_Valid
    ConvolutionTest.ConvolveF32BackwardInputGroupedConvolution
    Convolve_1x1x4x4_1x1x2x2_Valid/2.Types
    Convolve_1x1x4x4_1x1x2x2_Valid/1.Types
    Convolve_1x1x4x4_1x1x2x2_Same/1.Types
    Convolve_1x1x4x4_1x1x2x2_Same/2.Types
    Convolve_1x1x4x4_1x1x3x3_Same/1.Types
    Convolve_1x1x4x4_1x1x3x3_Same/2.Types
    Convolve2D*

    # @xla//xla/tests:convolution_1d_autotune_disabled_test
    ConvolutionTest.Convolve1D*
    Convolve1D_1x2x5_1x2x2*
    Convolve1D1WindowTest_Instantiation/Convolve1D1WindowTestFloat*
    Convolve1D1WindowTest_Instantiation/Convolve1D1WindowTestHalf*

    # @xla//xla/service/gpu/transforms:cudnn_fused_conv_rewriter_autotune_disabled_test_amdgpu_any
    CudnnFusedConvRewriterTest.TestRelu6
    CudnnFusedConvRewriterTest.TestLeakyRelu
    CudnnFusedConvRewriterTest.TestBiasMultiCall
    CudnnFusedConvRewriterTest.TestElu
    CudnnFusedConvRewriterTest.TestBias
    CudnnFusedConvRewriterTest.TestBiasNoRelu
    CudnnFusedConvRewriterTest.Test3D
    CudnnFusedConvRewriterTest.TestRelu6OddChannels
    CudnnFusedConvRewriterTest.TestConvOnly

    # @xla//xla/tools:xla_gpu_compile_lib_test_amdgpu_any
    XlaCompileLibTest.CompilesForGpuWithoutDevice

    # @xla//xla/tools:xla_deviceless_compile_lib_test 
    XlaDevicelessCompileLibTest/XlaDevicelessCompileLibTest.CompilesForGpuWithoutDevice/LegacyAotFlow
    XlaDevicelessCompileLibTest/XlaDevicelessCompileLibTest.CompilesForGpuWithoutDevice/NewAotFlow

    # @xla//xla/service/gpu/autotuning:gemm_fusion_autotuner_test_amdgpu_any
    GemmFusionAutotunerTest.Int8FusedGemm # failing on mi250
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
    -- @xla//xla/... \
    # ^^^ TODO (rocm) weekly-sync-20251021 excluded test files
