#include "tensorflow/stream_executor/cuda/cuda_rng.h"

#include <dlfcn.h>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/dso_loader.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/rng.h"
#include "third_party/gpus/cuda/include/curand.h"

// Formats curandStatus_t to output prettified values into a log stream.
std::ostream &operator<<(std::ostream &in, const curandStatus_t &status) {
#define OSTREAM_CURAND_STATUS(__name) \
  case CURAND_STATUS_##__name:        \
    in << "CURAND_STATUS_" #__name;   \
    return in;

  switch (status) {
    OSTREAM_CURAND_STATUS(SUCCESS)
    OSTREAM_CURAND_STATUS(VERSION_MISMATCH)
    OSTREAM_CURAND_STATUS(NOT_INITIALIZED)
    OSTREAM_CURAND_STATUS(ALLOCATION_FAILED)
    OSTREAM_CURAND_STATUS(TYPE_ERROR)
    OSTREAM_CURAND_STATUS(OUT_OF_RANGE)
    OSTREAM_CURAND_STATUS(LENGTH_NOT_MULTIPLE)
    OSTREAM_CURAND_STATUS(LAUNCH_FAILURE)
    OSTREAM_CURAND_STATUS(PREEXISTING_FAILURE)
    OSTREAM_CURAND_STATUS(INITIALIZATION_FAILED)
    OSTREAM_CURAND_STATUS(ARCH_MISMATCH)
    OSTREAM_CURAND_STATUS(INTERNAL_ERROR)
    default:
      in << "curandStatus_t(" << static_cast<int>(status) << ")";
      return in;
  }
}

namespace perftools {
namespace gputools {
namespace cuda {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuRandPlugin);

namespace dynload {

#define PERFTOOLS_GPUTOOLS_CURAND_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                            \
    static const char *kName;                                               \
    using FuncPointerT = std::add_pointer<decltype(::__name)>::type;        \
    static void *GetDsoHandle() {                                           \
      static auto status = internal::CachedDsoLoader::GetCurandDsoHandle(); \
      return status.ValueOrDie();                                           \
    }                                                                       \
    static FuncPointerT DynLoad() {                                         \
      static void *f = dlsym(GetDsoHandle(), kName);                        \
      CHECK(f != nullptr) << "could not find " << kName                     \
                          << " in curand DSO; dlerror: " << dlerror();      \
      return reinterpret_cast<FuncPointerT>(f);                             \
    }                                                                       \
    template <typename... Args>                                             \
    curandStatus_t operator()(CUDAExecutor * parent, Args... args) {        \
      cuda::ScopedActivateExecutorContext sac{parent};                      \
      return DynLoad()(args...);                                            \
    }                                                                       \
  } __name;                                                                 \
  const char *DynLoadShim__##__name::kName = #__name;

PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandCreateGenerator);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandDestroyGenerator);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandSetStream);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandGenerateUniform);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandGenerateUniformDouble);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandSetPseudoRandomGeneratorSeed);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandSetGeneratorOffset);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandGenerateNormal);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandGenerateNormalDouble);

}  // namespace dynload

template <typename T>
string TypeString();

template <>
string TypeString<float>() {
  return "float";
}

template <>
string TypeString<double>() {
  return "double";
}

template <>
string TypeString<std::complex<float>>() {
  return "std::complex<float>";
}

template <>
string TypeString<std::complex<double>>() {
  return "std::complex<double>";
}

CUDARng::CUDARng(CUDAExecutor *parent) : parent_(parent), rng_(nullptr) {}

CUDARng::~CUDARng() {
  if (rng_ != nullptr) {
    dynload::curandDestroyGenerator(parent_, rng_);
  }
}

bool CUDARng::Init() {
  mutex_lock lock{mu_};
  CHECK(rng_ == nullptr);

  curandStatus_t ret =
      dynload::curandCreateGenerator(parent_, &rng_, CURAND_RNG_PSEUDO_DEFAULT);
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create random number generator: " << ret;
    return false;
  }

  CHECK(rng_ != nullptr);
  return true;
}

bool CUDARng::SetStream(Stream *stream) {
  curandStatus_t ret =
      dynload::curandSetStream(parent_, rng_, AsCUDAStreamValue(stream));
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for random generation: " << ret;
    return false;
  }

  return true;
}

// Returns true if std::complex stores its contents as two consecutive
// elements. Tests int, float and double, as the last two are independent
// specializations.
constexpr bool ComplexIsConsecutiveFloats() {
  return sizeof(std::complex<int>) == 8 && sizeof(std::complex<float>) == 8 &&
      sizeof(std::complex<double>) == 16;
}

template <typename T>
bool CUDARng::DoPopulateRandUniformInternal(Stream *stream,
                                            DeviceMemory<T> *v) {
  mutex_lock lock{mu_};
  static_assert(ComplexIsConsecutiveFloats(),
                "std::complex values are not stored as consecutive values");

  if (!SetStream(stream)) {
    return false;
  }

  // std::complex<T> is currently implemented as two consecutive T variables.
  uint64 element_count = v->ElementCount();
  if (std::is_same<T, std::complex<float>>::value ||
      std::is_same<T, std::complex<double>>::value) {
    element_count *= 2;
  }

  curandStatus_t ret;
  if (std::is_same<T, float>::value ||
      std::is_same<T, std::complex<float>>::value) {
    ret = dynload::curandGenerateUniform(
        parent_, rng_, reinterpret_cast<float *>(CUDAMemoryMutable(v)),
        element_count);
  } else {
    ret = dynload::curandGenerateUniformDouble(
        parent_, rng_, reinterpret_cast<double *>(CUDAMemoryMutable(v)),
        element_count);
  }
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do uniform generation of " << v->ElementCount()
               << " " << TypeString<T>() << "s at " << v->opaque() << ": "
               << ret;
    return false;
  }

  return true;
}

bool CUDARng::DoPopulateRandUniform(Stream *stream, DeviceMemory<float> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool CUDARng::DoPopulateRandUniform(Stream *stream, DeviceMemory<double> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool CUDARng::DoPopulateRandUniform(Stream *stream,
                                    DeviceMemory<std::complex<float>> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool CUDARng::DoPopulateRandUniform(Stream *stream,
                                    DeviceMemory<std::complex<double>> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

template <typename ElemT, typename FuncT>
bool CUDARng::DoPopulateRandGaussianInternal(Stream *stream, ElemT mean,
                                             ElemT stddev,
                                             DeviceMemory<ElemT> *v,
                                             FuncT func) {
  mutex_lock lock{mu_};

  if (!SetStream(stream)) {
    return false;
  }

  uint64 element_count = v->ElementCount();
  curandStatus_t ret =
      func(parent_, rng_, CUDAMemoryMutable(v), element_count, mean, stddev);

  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do gaussian generation of " << v->ElementCount()
               << " floats at " << v->opaque() << ": " << ret;
    return false;
  }

  return true;
}

bool CUDARng::DoPopulateRandGaussian(Stream *stream, float mean, float stddev,
                                     DeviceMemory<float> *v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        dynload::curandGenerateNormal);
}

bool CUDARng::DoPopulateRandGaussian(Stream *stream, double mean, double stddev,
                                     DeviceMemory<double> *v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        dynload::curandGenerateNormalDouble);
}

bool CUDARng::SetSeed(Stream *stream, const uint8 *seed, uint64 seed_bytes) {
  mutex_lock lock{mu_};
  CHECK(rng_ != nullptr);

  if (!CheckSeed(seed, seed_bytes)) {
    return false;
  }

  if (!SetStream(stream)) {
    return false;
  }

  // Requires 8 bytes of seed data; checked in RngSupport::CheckSeed (above)
  // (which itself requires 16 for API consistency with host RNG fallbacks).
  curandStatus_t ret = dynload::curandSetPseudoRandomGeneratorSeed(
      parent_, rng_, *(reinterpret_cast<const uint64 *>(seed)));
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set rng seed: " << ret;
    return false;
  }

  ret = dynload::curandSetGeneratorOffset(parent_, rng_, 0);
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to reset rng position: " << ret;
    return false;
  }
  return true;
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

namespace gpu = ::perftools::gputools;

REGISTER_MODULE_INITIALIZER(register_curand, {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::RngFactory>(
              gpu::cuda::kCudaPlatformId, gpu::cuda::kCuRandPlugin, "cuRAND",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::rng::RngSupport * {
                gpu::cuda::CUDAExecutor *cuda_executor =
                    dynamic_cast<gpu::cuda::CUDAExecutor *>(parent);
                if (cuda_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the cuRAND "
                      << "support library with a non-CUDA StreamExecutor";
                  return nullptr;
                }

                gpu::cuda::CUDARng *rng = new gpu::cuda::CUDARng(cuda_executor);
                if (!rng->Init()) {
                  // Note: Init() will log a more specific error.
                  delete rng;
                  return nullptr;
                }
                return rng;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuRAND factory: "
               << status.error_message();
  }

  // Prime the cuRAND DSO. The loader will log more information.
  auto statusor = gpu::internal::CachedDsoLoader::GetCurandDsoHandle();
  if (!statusor.ok()) {
    LOG(INFO) << "Unable to load cuRAND DSO.";
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kRng,
                                                     gpu::cuda::kCuRandPlugin);
});
