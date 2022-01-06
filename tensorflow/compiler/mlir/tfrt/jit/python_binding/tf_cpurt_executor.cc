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

#include "tensorflow/compiler/mlir/tfrt/jit/python_binding/tf_cpurt_executor.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_cpurt_pipeline.h"
#include "tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs_registration.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tfrt/cpu/jit/cpurt.h"  // from @tf_runtime
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
using ::tfrt::DType;
using ::tfrt::ExecutionContext;
using ::tfrt::GetDType;
using ::tfrt::RCReference;
using ::tfrt::RemainingResults;
using ::tfrt::RequestContext;
using ::tfrt::RequestContextBuilder;
using ::tfrt::StrCat;
using ::tfrt::cpu::jit::CompilationOptions;
using ::tfrt::cpu::jit::Executable;
using ::tfrt::cpu::jit::JitExecutable;
using ::tfrt::cpu::jit::MemrefDesc;
using ::tfrt::cpu::jit::ReturnStridedMemref;
using ::tfrt::cpu::jit::ReturnValueConverter;

namespace tensorflow {

TfCpurtExecutor::TfCpurtExecutor()
    : host_context_(
          [](const DecodedDiagnostic& diag) {
            llvm::errs() << "Encountered runtime error: " << diag.message
                         << "\n";
          },
          CreateMallocAllocator(), CreateMultiThreadedWorkQueue(4, 4)) {}

TfCpurtExecutor::Handle TfCpurtExecutor::Compile(const std::string& mlir_module,
                                                 const std::string& entrypoint,
                                                 Specialization specialization,
                                                 bool vectorize,
                                                 bool legalize_i1_tensors) {
  CompilationOptions opts;
  // Create an async task for each worker thread.
  opts.num_worker_threads = 4;
  opts.register_dialects = [](mlir::DialectRegistry& registry) {
    mlir::RegisterAllTensorFlowDialects(registry);
    // Needed to verify function argument attributes which are used to
    // annotate dynamic shaped types with static type information.
    mlir::tfrt::RegisterPythonTestAttrsDialect(registry);
  };
  opts.register_pass_pipeline = [=](mlir::OpPassManager& pm) {
    tensorflow::TfCpuRtPipelineOptions opts;
    opts.vectorize = vectorize;
    opts.legalize_i1_tensors = legalize_i1_tensors;
    tensorflow::CreateTfCpuRtPipeline(pm, opts);
  };
  opts.specialization = specialization;
  opts.type_converter = mlir::bufferization::BufferizeTypeConverter();

  // Instantiate new JitExecutable from the MLIR source.
  llvm::Expected<JitExecutable> jit_executable =
      JitExecutable::Instantiate(mlir_module, entrypoint, opts);
  if (auto err = jit_executable.takeError())
    throw std::runtime_error(
        StrCat("Failed to instantiate JitExecutable: ", err));

  Handle hdl = jit_executables_.size();
  jit_executables_.insert({hdl, std::move(*jit_executable)});
  return hdl;
}

// Returns Python buffer protocol's type string from TFRT's dtype.
static const char* ToPythonStructFormat(DType dtype_kind) {
  // Reference: https://docs.python.org/3/library/struct.html

  switch (dtype_kind) {
    case DType::Invalid:
      throw std::runtime_error("Invalid dtype.");
    case DType::Unsupported:
      throw std::runtime_error("Unsupported dtype.");
    case DType::UI8:
      return "B";
    case DType::UI16:
      return "H";
    case DType::UI32:
      return "I";
    case DType::UI64:
      return "Q";
    case DType::I1:
      return "?";
    case DType::I8:
      return "b";
    case DType::I16:
      return "h";
    case DType::I32:
      return "i";
    case DType::I64:
      return "q";
    case DType::F32:
      return "f";
    case DType::F64:
      return "d";
    case DType::Complex64:
      throw std::runtime_error("Unimplemented.");
    case DType::Complex128:
      throw std::runtime_error("Unimplemented.");
    case DType::F16:
      throw std::runtime_error("Unimplemented.");
    case DType::BF16:
      throw std::runtime_error("Unimplemented.");
    case DType::String:
      throw std::runtime_error("Unimplemented.");
    default:
      throw std::runtime_error("Unimplemented.");
  }
}

// Returns TFRT's dtype for the Python buffer protocol's type string.
static DType FromPythonStructFormat(char dtype) {
  // Reference: https://docs.python.org/3/library/struct.html
  switch (dtype) {
    case 'B':
      return DType::UI8;
    case 'H':
      return DType::UI16;
    case 'I':
      return DType::UI32;
    case 'Q':
      return DType::UI64;
    case '?':
      return DType::I1;
    case 'b':
      return DType::I8;
    case 'h':
      return DType::I16;
    case 'i':
      return DType::I32;
    case 'q':
      return DType::I64;
    case 'f':
      return DType::F32;
    case 'd':
      return DType::F64;
    default:
      throw std::runtime_error("Unsupported python dtype.");
  }
}

// Converts Python array to the Memref Descriptor.
static void ConvertPyArrayMemrefDesc(const py::array& array,
                                     MemrefDesc* memref) {
  auto py_dtype = [](pybind11::dtype dtype) -> char {
    // np.int64 array for some reason has `i` dtype, however according to the
    // documentation it must be `q`.
    if (dtype.kind() == 'i' && dtype.itemsize() == 8) return 'q';

    return dtype.char_();
  };

  memref->dtype = DType(FromPythonStructFormat(py_dtype(array.dtype())));
  memref->data = const_cast<void*>(array.data());
  memref->offset = 0;

  int rank = array.ndim();
  memref->sizes.resize(rank);
  memref->strides.resize(rank);

  for (ssize_t d = 0; d < rank; ++d) {
    memref->sizes[d] = array.shape(d);
    memref->strides[d] = array.strides(d) / array.itemsize();
  }
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

using PyBindingReturnValueConverter =
    ReturnValueConverter<PyBindingConversionContext>;
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
// ReturnStridedMemref's concept (see cpurt.h).
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

std::vector<py::array> TfCpurtExecutor::Execute(
    Handle handle, const std::vector<py::array>& arguments) {
  // Verify that we have a compilatio result for the handle.
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
  std::vector<MemrefDesc> memrefs(arguments.size());
  for (int i = 0; i < arguments.size(); ++i)
    ConvertPyArrayMemrefDesc(arguments[i], &memrefs[i]);

  // Get an executable that might be specialized to the operands.
  llvm::Expected<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(memrefs, exec_ctx);
  if (auto err = executable.takeError())
    throw std::runtime_error(
        StrCat("Failed to get Executable: ", std::move(err)));

  // Wait for the compilation completion.
  host_context_.Await({executable->CopyRef()});

  if (executable->IsError())
    throw std::runtime_error(
        StrCat("Failed to get Executable: ", executable->GetError()));

  // Prepare storage for returned values.
  unsigned num_results = (*executable)->num_results();
  std::vector<RCReference<AsyncValue>> result_storage(num_results);

  RemainingResults results(result_storage);

  // Convert returned memrefs to Tensors.
  PyBindingReturnValueConverter converter(results);
  converter.AddConversion(ReturnStridedMemref<MemrefToPyArray>);
  if (auto err = (*executable)->Execute(memrefs, converter, exec_ctx))
    throw std::runtime_error(StrCat("Unsupported argument: ", err));

  // Pull Python arrays out of async values.
  std::vector<py::array> ret_values;
  ret_values.reserve(result_storage.size());
  for (auto& result : result_storage) {
    if (result->IsError())
      throw std::runtime_error(StrCat("result error: ", result->GetError()));
    py::array& result_array = result->get<py::array>();
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(result_array.data(),
                                      result_array.nbytes());
    ret_values.emplace_back(result_array);
  }

  return ret_values;
}

bool TfCpurtExecutor::BuiltWith(const std::string& cpu_feature) {
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

PYBIND11_MODULE(_tf_cpurt_executor, m) {
  py::enum_<tensorflow::TfCpurtExecutor::Specialization>(m, "Specialization")
      .value("ENABLED", tensorflow::TfCpurtExecutor::Specialization::kEnabled)
      .value("DISABLED", tensorflow::TfCpurtExecutor::Specialization::kDisabled)
      .value("ALWAYS", tensorflow::TfCpurtExecutor::Specialization::kAlways);

  py::class_<tensorflow::TfCpurtExecutor>(m, "TfCpurtExecutor")
      .def(py::init<>())
      .def("compile", &tensorflow::TfCpurtExecutor::Compile,
           py::arg("mlir_module"), py::arg("entrypoint"),
           py::arg("specialization") =
               tensorflow::TfCpurtExecutor::Specialization::kEnabled,
           py::arg("vectorize") = false, py::arg("legalize_i1_tensors") = false)
      .def("execute", &tensorflow::TfCpurtExecutor::Execute)
      .def("built_with", &tensorflow::TfCpurtExecutor::BuiltWith,
           py::arg("cpu_feature"));
}
