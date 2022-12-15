/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_jitrt_executor.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/jit/python_binding/conversion_utils.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.h"
#include "tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs_registration.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compiler.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tfrt/jitrt/async_task_runner.h"  // from @tf_runtime
#include "tfrt/jitrt/jitrt_compiler.h"  // from @tf_runtime
#include "tfrt/jitrt/results.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace py = pybind11;

using ::tfrt::AsyncValue;
using ::tfrt::AsyncValuePtr;
using ::tfrt::CreateMallocAllocator;
using ::tfrt::CreateMultiThreadedWorkQueue;
using ::tfrt::DecodedDiagnostic;
using ::tfrt::GetDType;
using ::tfrt::RCReference;
using ::tfrt::RemainingResults;
using ::tfrt::RequestContext;
using ::tfrt::RequestContextBuilder;
using ::tfrt::StrCat;

using ::tfrt::jitrt::CompilationPipelineOptions;
using ::tfrt::jitrt::CreateDefaultJitRtCompilationPipeline;
using ::tfrt::jitrt::HostContextAsyncTaskRunner;
using ::tfrt::jitrt::RegisterDefaultJitRtDialects;
using ::tfrt::jitrt::RemainingResultsConverter;
using ::tfrt::jitrt::ReturnStridedMemref;

using ::xla::runtime::Executable;
using ::xla::runtime::JitExecutable;
using ::xla::runtime::MemrefDesc;

namespace tensorflow {

TfJitRtExecutor::TfJitRtExecutor()
    : host_context_(
          [](const DecodedDiagnostic& diag) {
            llvm::errs() << "Encountered runtime error: " << diag.message()
                         << "\n";
          },
          CreateMallocAllocator(), CreateMultiThreadedWorkQueue(4, 4)) {}

TfJitRtExecutor::Handle TfJitRtExecutor::Compile(
    const std::string& mlir_module, const std::string& entrypoint,
    Specialization specialization, bool vectorize, bool codegen_transpose,
    bool legalize_i1_tensors, bool peel, bool enable_xla_cpu_transformations,
    bool pack_matmul) {
  // Options for the default JitRt compilation pipeline (lowering to LLVM).
  CompilationPipelineOptions copts;
  copts.alignment = EIGEN_MAX_ALIGN_BYTES;
  copts.num_worker_threads = 4;

  JitExecutable::Options opts;
  opts.compiler.register_dialects =
      [](xla::runtime::DialectRegistry& dialects) {
        mlir::RegisterAllTensorFlowDialects(*dialects);
        RegisterDefaultJitRtDialects(dialects);
        // Needed to verify function argument attributes which are used to
        // annotate dynamic shaped types with static type information.
        mlir::tfrt::RegisterPythonTestAttrsDialect(*dialects);
      };
  opts.compiler.create_compilation_pipeline =
      [=](xla::runtime::PassManager& passes) {
        tensorflow::TfJitRtPipelineOptions opts;
        opts.vectorize = vectorize;
        opts.codegen_transpose = codegen_transpose;
        opts.legalize_i1_tensors = legalize_i1_tensors;
        opts.peel = peel;
        opts.enable_xla_cpu_transformations = enable_xla_cpu_transformations;
        opts.lower_to_mmt4d = pack_matmul;
        tensorflow::CreateTfJitRtPipeline(*passes, opts);
        CreateDefaultJitRtCompilationPipeline(passes, copts);
      };
  if (specialization != Specialization::kDisabled) {
    opts.compiler.create_specialization_pipeline =
        CreateJitRtSpecializationPipeline;
  }
  opts.specialization = specialization;
  opts.compiler.calling_convention = xla::runtime::DefaultCallingConvention(
      mlir::bufferization::BufferizeTypeConverter());

  // Instantiate new JitExecutable from the MLIR source.
  absl::StatusOr<JitExecutable> jit_executable =
      JitExecutable::Instantiate(mlir_module, entrypoint, opts);
  if (!jit_executable.ok())
    throw std::runtime_error(StrCat("Failed to instantiate JitExecutable: ",
                                    jit_executable.status().message()));

  Handle hdl = jit_executables_.size();
  jit_executables_.insert({hdl, std::move(*jit_executable)});
  return hdl;
}

template <typename T, int rank>
static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
  return memref->sizes;
}

template <typename T, int rank>
static llvm::ArrayRef<int64_t> Strides(StridedMemRefType<T, rank>* memref) {
  return memref->strides;
}

template <typename T>
static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
  return {};
}

template <typename T>
static llvm::ArrayRef<int64_t> Strides(StridedMemRefType<T, 0>* memref) {
  return {};
}

namespace {
struct PyBindingConversionContext {};

using PyBindingResultConverter =
    RemainingResultsConverter<PyBindingConversionContext>;
}  // namespace

template <typename T>
static bool IsAligned(const T* ptr) {
#if EIGEN_MAX_ALIGN_BYTES == 0
  return true;
#else
  return reinterpret_cast<intptr_t>(ptr) % EIGEN_MAX_ALIGN_BYTES == 0;
#endif
}

// Converts StridedMemrefType to the Python array. This struct satisfies
// ReturnStridedMemref's concept (see jitrt.h).
//
// TODO(ezhulenev): Currently this converter transfers ownership of the memref
// to the Python array. This is not correct in general, because memref does not
// imply ownership, for example it can be one of the forwarded inputs or a
// global memref that is owned by the compiled kernel.
struct MemrefToPyArray {
  using ResultType = py::array;
  using ConversionContext = PyBindingConversionContext;

  template <typename T, int rank>
  static py::array Convert(const ConversionContext&, void* memref_ptr) {
    auto* memref = static_cast<StridedMemRefType<T, rank>*>(memref_ptr);
    assert(IsAligned(memref->data) && "returned memref must be aligned");

    auto memref_sizes = Sizes(memref);
    auto memref_strides = Strides(memref);

    std::vector<ssize_t> sizes(memref_sizes.begin(), memref_sizes.end());
    std::vector<ssize_t> strides(memref_strides.begin(), memref_strides.end());

    // Python expects strides in bytes.
    auto dtype = GetDType<T>();
    for (size_t d = 0; d < strides.size(); ++d)
      strides[d] *= GetHostSize(dtype);

    return py::array(py::buffer_info(memref->data, GetHostSize(dtype),
                                     ToPythonStructFormat(dtype), rank, sizes,
                                     strides));
  }
};

std::vector<py::array> TfJitRtExecutor::Execute(
    Handle handle, const std::vector<py::array>& arguments) {
  // Verify that we have a compilation result for the handle.
  auto it = jit_executables_.find(handle);
  if (it == jit_executables_.end())
    throw std::runtime_error(StrCat("Unknown jit executable handle: ", handle));

  JitExecutable& jit_executable = it->getSecond();

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(&host_context_, /*resource_context=*/nullptr)
          .build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  // Convert arguments to memrefs.
  std::vector<MemrefDesc> memrefs;
  memrefs.reserve(arguments.size());
  for (int i = 0; i < arguments.size(); ++i)
    memrefs.emplace_back(ConvertPyArrayMemrefDesc(arguments[i]));

  // Get an executable that might be specialized to the operands.
  absl::StatusOr<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(memrefs);
  if (!executable.ok())
    throw std::runtime_error(
        StrCat("Failed to get Executable: ", executable.status().message()));

  // Wait for the compilation completion.
  host_context_.Await({executable->CopyRef()});

  if (executable->IsError())
    throw std::runtime_error(
        StrCat("Failed to get Executable: ", executable->GetError().message()));

  // Prepare storage for returned values.
  unsigned num_results = (*executable)->num_results();
  std::vector<RCReference<AsyncValue>> result_storage(num_results);

  RemainingResults results(result_storage);

  // Execute async tasks in the HostContext work queue.
  Executable::ExecuteOpts opts;
  HostContextAsyncTaskRunner async_task_runner(&host_context_);
  opts.async_task_runner = &async_task_runner;

  // Convert returned memrefs to python arrays.
  PyBindingConversionContext results_ctx;
  PyBindingResultConverter converter(results, results_ctx);
  converter.AddConversion(ReturnStridedMemref<MemrefToPyArray>);
  if (auto st = (*executable)->Execute(memrefs, converter, opts); !st.ok())
    throw std::runtime_error(StrCat("Unsupported argument: ", st.message()));

  // Pull Python arrays out of async values.
  std::vector<py::array> ret_values;
  ret_values.reserve(result_storage.size());
  for (auto& result : result_storage) {
    if (result->IsError())
      throw std::runtime_error(
          StrCat("result error: ", result->GetError().message()));
    py::array& result_array = result->get<py::array>();
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(result_array.data(),
                                      result_array.nbytes());
    ret_values.emplace_back(result_array);
  }

  return ret_values;
}

bool TfJitRtExecutor::BuiltWith(const std::string& cpu_feature) {
  if (cpu_feature == "AVX2") {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
  }
  return false;
}

}  // namespace tensorflow

PYBIND11_MODULE(_tf_jitrt_executor, m) {
  py::enum_<tensorflow::TfJitRtExecutor::Specialization>(m, "Specialization")
      .value("ENABLED", tensorflow::TfJitRtExecutor::Specialization::kEnabled)
      .value("DISABLED", tensorflow::TfJitRtExecutor::Specialization::kDisabled)
      .value("ALWAYS", tensorflow::TfJitRtExecutor::Specialization::kAlways);

  py::class_<tensorflow::TfJitRtExecutor>(m, "TfJitRtExecutor")
      .def(py::init<>())
      .def("compile", &tensorflow::TfJitRtExecutor::Compile,
           py::arg("mlir_module"), py::arg("entrypoint"),
           py::arg("specialization") =
               tensorflow::TfJitRtExecutor::Specialization::kEnabled,
           py::arg("vectorize") = false, py::arg("codegen_transpose") = false,
           py::arg("legalize_i1_tensors") = false, py::arg("peel") = true,
           py::arg("enable_xla_cpu_transformations") = false,
           py::arg("pack_matmul") = false)
      .def("execute", &tensorflow::TfJitRtExecutor::Execute)
      .def("built_with", &tensorflow::TfJitRtExecutor::BuiltWith,
           py::arg("cpu_feature"));
}
