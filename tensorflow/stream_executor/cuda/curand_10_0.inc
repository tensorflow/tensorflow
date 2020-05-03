// Auto-generated, do not edit.

extern "C" {
curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t *generator,
                                               curandRngType_t rng_type) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(curandGenerator_t *, curandRngType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandCreateGenerator");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, rng_type);
}

curandStatus_t CURANDAPI curandCreateGeneratorHost(curandGenerator_t *generator,
                                                   curandRngType_t rng_type) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(curandGenerator_t *, curandRngType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandCreateGeneratorHost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, rng_type);
}

curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t generator) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandDestroyGenerator");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator);
}

curandStatus_t CURANDAPI curandGetVersion(int *version) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGetVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(version);
}

curandStatus_t CURANDAPI curandGetProperty(libraryPropertyType type,
                                           int *value) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(libraryPropertyType, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGetProperty");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(type, value);
}

curandStatus_t CURANDAPI curandSetStream(curandGenerator_t generator,
                                         cudaStream_t stream) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandSetStream");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, stream);
}

curandStatus_t CURANDAPI curandSetPseudoRandomGeneratorSeed(
    curandGenerator_t generator, unsigned long long seed) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(curandGenerator_t, unsigned long long);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("curandSetPseudoRandomGeneratorSeed");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, seed);
}

curandStatus_t CURANDAPI curandSetGeneratorOffset(curandGenerator_t generator,
                                                  unsigned long long offset) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(curandGenerator_t, unsigned long long);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandSetGeneratorOffset");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, offset);
}

curandStatus_t CURANDAPI curandSetGeneratorOrdering(curandGenerator_t generator,
                                                    curandOrdering_t order) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(curandGenerator_t, curandOrdering_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandSetGeneratorOrdering");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, order);
}

curandStatus_t CURANDAPI curandSetQuasiRandomGeneratorDimensions(
    curandGenerator_t generator, unsigned int num_dimensions) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t, unsigned int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("curandSetQuasiRandomGeneratorDimensions");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, num_dimensions);
}

curandStatus_t CURANDAPI curandGenerate(curandGenerator_t generator,
                                        unsigned int *outputPtr, size_t num) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(curandGenerator_t, unsigned int *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, num);
}

curandStatus_t CURANDAPI curandGenerateLongLong(curandGenerator_t generator,
                                                unsigned long long *outputPtr,
                                                size_t num) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t,
                                              unsigned long long *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateLongLong");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, num);
}

curandStatus_t CURANDAPI curandGenerateUniform(curandGenerator_t generator,
                                               float *outputPtr, size_t num) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(curandGenerator_t, float *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateUniform");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, num);
}

curandStatus_t CURANDAPI curandGenerateUniformDouble(
    curandGenerator_t generator, double *outputPtr, size_t num) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(curandGenerator_t, double *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateUniformDouble");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, num);
}

curandStatus_t CURANDAPI curandGenerateNormal(curandGenerator_t generator,
                                              float *outputPtr, size_t n,
                                              float mean, float stddev) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t, float *,
                                              size_t, float, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateNormal");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, n, mean, stddev);
}

curandStatus_t CURANDAPI curandGenerateNormalDouble(curandGenerator_t generator,
                                                    double *outputPtr, size_t n,
                                                    double mean,
                                                    double stddev) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t, double *,
                                              size_t, double, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateNormalDouble");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, n, mean, stddev);
}

curandStatus_t CURANDAPI curandGenerateLogNormal(curandGenerator_t generator,
                                                 float *outputPtr, size_t n,
                                                 float mean, float stddev) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t, float *,
                                              size_t, float, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateLogNormal");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, n, mean, stddev);
}

curandStatus_t CURANDAPI
curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr,
                              size_t n, double mean, double stddev) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t, double *,
                                              size_t, double, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateLogNormalDouble");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, n, mean, stddev);
}

curandStatus_t CURANDAPI curandCreatePoissonDistribution(
    double lambda, curandDiscreteDistribution_t *discrete_distribution) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(double, curandDiscreteDistribution_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandCreatePoissonDistribution");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lambda, discrete_distribution);
}

curandStatus_t CURANDAPI
curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandDiscreteDistribution_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandDestroyDistribution");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(discrete_distribution);
}

curandStatus_t CURANDAPI curandGeneratePoisson(curandGenerator_t generator,
                                               unsigned int *outputPtr,
                                               size_t n, double lambda) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t, unsigned int *,
                                              size_t, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGeneratePoisson");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, n, lambda);
}

curandStatus_t CURANDAPI curandGeneratePoissonMethod(
    curandGenerator_t generator, unsigned int *outputPtr, size_t n,
    double lambda, curandMethod_t method) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t, unsigned int *,
                                              size_t, double, curandMethod_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGeneratePoissonMethod");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, n, lambda, method);
}

curandStatus_t CURANDAPI curandGenerateBinomial(curandGenerator_t generator,
                                                unsigned int *outputPtr,
                                                size_t num, unsigned int n,
                                                double p) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t, unsigned int *,
                                              size_t, unsigned int, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateBinomial");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, num, n, p);
}

curandStatus_t CURANDAPI curandGenerateBinomialMethod(
    curandGenerator_t generator, unsigned int *outputPtr, size_t num,
    unsigned int n, double p, curandMethod_t method) {
  using FuncPtr =
      curandStatus_t(CURANDAPI *)(curandGenerator_t, unsigned int *, size_t,
                                  unsigned int, double, curandMethod_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateBinomialMethod");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator, outputPtr, num, n, p, method);
}

curandStatus_t CURANDAPI curandGenerateSeeds(curandGenerator_t generator) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandGenerator_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGenerateSeeds");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(generator);
}

curandStatus_t CURANDAPI curandGetDirectionVectors32(
    curandDirectionVectors32_t *vectors[], curandDirectionVectorSet_t set) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandDirectionVectors32_t *[],
                                              curandDirectionVectorSet_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGetDirectionVectors32");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(vectors, set);
}

curandStatus_t CURANDAPI
curandGetScrambleConstants32(unsigned int **constants) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(unsigned int **);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGetScrambleConstants32");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(constants);
}

curandStatus_t CURANDAPI curandGetDirectionVectors64(
    curandDirectionVectors64_t *vectors[], curandDirectionVectorSet_t set) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(curandDirectionVectors64_t *[],
                                              curandDirectionVectorSet_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGetDirectionVectors64");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(vectors, set);
}

curandStatus_t CURANDAPI
curandGetScrambleConstants64(unsigned long long **constants) {
  using FuncPtr = curandStatus_t(CURANDAPI *)(unsigned long long **);
  static auto func_ptr = LoadSymbol<FuncPtr>("curandGetScrambleConstants64");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(constants);
}

}  // extern "C"
