/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/runtime/symbolic_shape.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/constraints.h"
#include "tensorflow/compiler/xla/runtime/types.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/test_benchmark.h"

namespace xla {
namespace runtime {

using llvm::ArrayRef;

using SymbolicShape = SymbolicShapesResolver::SymbolicShape;

// Create a function type with empty results from the operands shapes.
static FunctionType GetFunctionType(
    llvm::SmallVector<PrimitiveType> dtypes,
    llvm::SmallVector<std::optional<SymbolicShape>> shapes) {
  std::vector<std::unique_ptr<Type>> operands;
  operands.reserve(shapes.size());

  for (auto tuple : llvm::zip(dtypes, shapes)) {
    auto dtype = std::get<0>(tuple);
    auto shape = std::get<1>(tuple);
    if (shape.has_value()) {
      operands.push_back(std::make_unique<MemrefType>(*shape, dtype));
    } else {
      operands.push_back(std::make_unique<UnrankedMemrefType>(dtype));
    }
  }

  return FunctionType(std::move(operands), {});
}

// Creates fake opaque argument.
static OpaqueArg GetFakeOpaqueArg() { return OpaqueArg(nullptr); }

// Creates fake memref argument of the given shape.
static MemrefDesc GetFakeMemref(SymbolicShape shape) {
  // Data type of the fake memrefs doesn't matter.
  return MemrefDesc(PrimitiveType::F32, nullptr, 0, shape,
                    shape /* fake strides */);
}

// Creates fake memref arguments of the given shapes.
static llvm::SmallVector<MemrefDesc> GetFakeMemrefs(
    llvm::SmallVector<SymbolicShape> shapes) {
  llvm::SmallVector<MemrefDesc> memrefs;
  memrefs.reserve(shapes.size());
  for (auto& shape : shapes) memrefs.push_back(GetFakeMemref(shape));
  return memrefs;
}

// A helper function to convert initializer list to a list of shapes.
static llvm::SmallVector<SymbolicShape> SymbolicShapes(
    llvm::SmallVector<SymbolicShape> shapes) {
  return shapes;
}

TEST(SymbolicShapeResolverTest, UnrankedInputs) {
  // Operands: tensor<*xf32>, tensor<?xi32>, tensor<?x4xi1>
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::PRED};

  auto type = GetFunctionType(
      dtypes,
      {std::nullopt, {{MemrefType::kDynamic}}, {{MemrefType::kDynamic, 4}}});

  auto constraints = {ArgumentConstraint::kResolved,
                      ArgumentConstraint::kResolved,
                      ArgumentConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown dimensions are the same at runtime.
    auto operands = GetFakeMemrefs({{100, 100}, {100}, {100, 4}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, -2}, {-2}, {-2, 4}}));

    llvm::SmallVector<int64_t> values = {2, -2, -2, -2, -2, 4};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // All unknown dimensions are unique at runtime.
    auto operands = GetFakeMemrefs({{100, 101}, {102}, {103, 4}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, -3}, {-4}, {-5, 4}}));

    llvm::SmallVector<int64_t> values = {2, -2, -3, -4, -5, 4};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // Ones converted to a static dimension.
    auto operands = GetFakeMemrefs({{1, 1, 1}, {1}, {1, 4}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{1, 1, 1}, {1}, {1, 4}}));

    llvm::SmallVector<int64_t> values = {3, 1, 1, 1, 1, 1, 4};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // Known constants converted to a static dimension.
    auto operands = GetFakeMemrefs({{100, 4}, {4}, {1, 4}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {4}, {1, 4}}));

    llvm::SmallVector<int64_t> values = {2, -2, 4, 4, 1, 4};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }
}

TEST(SymbolicShapeResolverTest, DynamicInputShapes) {
  // Operands: tensor<?xf32>, tensor<?xi32>, tensor<?xi1>
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::PRED};
  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamic}},
                                       {{MemrefType::kDynamic}},
                                       {{MemrefType::kDynamic}}});

  auto constraints = {ArgumentConstraint::kResolved,
                      ArgumentConstraint::kResolved,
                      ArgumentConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown dimensions are the same at runtime.
    auto operands = GetFakeMemrefs({{100}, {100}, {100}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2}, {-2}, {-2}}));

    llvm::SmallVector<int64_t> values = {-2, -2, -2};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // All unknown dimensions are unique at runtime.
    auto operands = GetFakeMemrefs({{100}, {101}, {102}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2}, {-3}, {-4}}));

    llvm::SmallVector<int64_t> values = {-2, -3, -4};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // Two of the three dimensions are the same.
    auto operands = GetFakeMemrefs({{100}, {101}, {100}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2}, {-3}, {-2}}));

    llvm::SmallVector<int64_t> values = {-2, -3, -2};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // Ones converted to a static dimension.
    auto operands = GetFakeMemrefs({{1}, {1}, {100}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{1}, {1}, {-2}}));

    llvm::SmallVector<int64_t> values = {1, 1, -2};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }
}

TEST(SymbolicShapeResolverTest, PartialInputShapes) {
  // Operands: tensor<?x4xf32>, tensor<?x8xi32>, tensor<?xi1>
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::PRED};
  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamic, 4}},
                                       {{MemrefType::kDynamic, 8}},
                                       {{MemrefType::kDynamic}}});

  auto constraints = {ArgumentConstraint::kResolved,
                      ArgumentConstraint::kResolved,
                      ArgumentConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown dimensions are the same at runtime.
    auto operands = GetFakeMemrefs({{100, 4}, {100, 8}, {100}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {-2, 8}, {-2}}));

    llvm::SmallVector<int64_t> values = {-2, 4, -2, 8, -2};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // All unknown dimensions are unique at runtime.
    auto operands = GetFakeMemrefs({{100, 4}, {101, 8}, {102}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {-3, 8}, {-4}}));

    llvm::SmallVector<int64_t> values = {-2, 4, -3, 8, -4};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // Two of the three dimensions are the same.
    auto operands = GetFakeMemrefs({{100, 4}, {101, 8}, {100}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {-3, 8}, {-2}}));

    llvm::SmallVector<int64_t> values = {-2, 4, -3, 8, -2};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // Ones converted to a static dimension.
    auto operands = GetFakeMemrefs({{1, 4}, {100, 8}, {1}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{1, 4}, {-2, 8}, {1}}));

    llvm::SmallVector<int64_t> values = {1, 4, -2, 8, 1};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // Known constants converted to a static dimension.
    auto operands = GetFakeMemrefs({{100, 4}, {8, 8}, {8}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {8, 8}, {8}}));

    llvm::SmallVector<int64_t> values = {-2, 4, 8, 8, 8};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }
}

TEST(SymbolicShapeResolverTest, ShapeConstrainedInput) {
  // Operands: tensor<*xf32>, tensor<?x4xi32>
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32};

  auto type =
      GetFunctionType(dtypes, {std::nullopt, {{MemrefType::kDynamic, 4}}});

  auto constraints = {ArgumentConstraint::kShape, ArgumentConstraint::kShape};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown materialized as static shapes.
    auto operands = GetFakeMemrefs({{100, 100}, {100, 4}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 2);
    EXPECT_EQ(*symbolic, SymbolicShapes({{100, 100}, {100, 4}}));

    llvm::SmallVector<int64_t> values = {100, 100, 100, 4};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }
}

TEST(SymbolicShapeResolverTest, ShapeConstrainedInputAfterDynamicInput) {
  // Operands: tensor<?x?xf32>, tensor<?x?xi32>
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32};

  auto type =
      GetFunctionType(dtypes, {{{MemrefType::kDynamic, MemrefType::kDynamic}},
                               {{MemrefType::kDynamic, MemrefType::kDynamic}}});

  auto constraints = {ArgumentConstraint::kResolved,
                      ArgumentConstraint::kShape};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown dimensions materialized as static shapes (first operand
     // resolved based on seen static shapes of the second one).
    auto operands = GetFakeMemrefs({{100, 50}, {100, 50}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 2);
    EXPECT_EQ(*symbolic, SymbolicShapes({{100, 50}, {100, 50}}));

    llvm::SmallVector<int64_t> values = {100, 50, 100, 50};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // Unknown dimension correctly resolved to a symbolic dimension.
    auto operands = GetFakeMemrefs({{100, 50}, {100, 4}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 2);
    EXPECT_EQ(*symbolic, SymbolicShapes({{100, -2}, {100, 4}}));

    llvm::SmallVector<int64_t> values = {100, 4, 100, -2};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }
}

TEST(SymbolicShapeResolverTest, StaticShapeOperandHash) {
  // Operands: tensor<?x?xf32>, tensor<4x4xi32>
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32};

  auto type = GetFunctionType(
      dtypes, {{{MemrefType::kDynamic, MemrefType::kDynamic}}, {{4, 4}}});

  auto constraints = {ArgumentConstraint::kResolved,
                      ArgumentConstraint::kShape};

  SymbolicShapesResolver resolver(type, constraints);

  {  // Static shape doesn't participate in the hash value.
    auto operands = GetFakeMemrefs({{2, 2}, {4, 4}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_EQ(symbolic->size(), 2);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, -2}, {4, 4}}));

    llvm::SmallVector<int64_t> values = {-2, -2};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }
}

TEST(SymbolicShapeResolverTest, IncompatibleInput) {
  // Operands: tensor<?x4xi32>
  auto dtypes = {PrimitiveType::F32};
  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamic, 4}}});
  auto constraints = {ArgumentConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  {  // Operand of a different rank.
    auto operands = GetFakeMemrefs({{100, 100, 100}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_FALSE(symbolic.ok());
    EXPECT_FALSE(hash.ok());
  }

  {  // Operand with mismatched static shape.
    auto operands = GetFakeMemrefs({{100, 100}});
    auto symbolic = resolver.Resolve(operands);
    auto hash = resolver.ResolveHash(operands);

    EXPECT_FALSE(symbolic.ok());
    EXPECT_FALSE(hash.ok());
  }
}

TEST(SymbolicShapeResolverTest, OpaqueAndShapedInputs) {
  std::vector<int64_t> shape = {MemrefType::kDynamic, 4};

  // Operands: !async.token, tensor<?x4xf32>, tensor<?x4xf32>
  std::vector<std::unique_ptr<Type>> operands;
  operands.push_back(std::make_unique<AsyncTokenType>());
  operands.push_back(std::make_unique<MemrefType>(shape, PrimitiveType::F32));
  operands.push_back(std::make_unique<MemrefType>(shape, PrimitiveType::F32));

  auto constraints = {ArgumentConstraint::kResolved,
                      ArgumentConstraint::kResolved,
                      ArgumentConstraint::kResolved};

  FunctionType type(std::move(operands), {});

  SymbolicShapesResolver resolver(type, constraints);

  {  // Operand of a different shape.
    Arguments<OpaqueArg, MemrefDesc> arguments(3);
    arguments.push_back(GetFakeOpaqueArg());
    arguments.push_back(GetFakeMemref({2, 4}));
    arguments.push_back(GetFakeMemref({3, 4}));

    auto symbolic = resolver.Resolve(arguments);
    auto hash = resolver.ResolveHash(arguments);

    ASSERT_TRUE(symbolic.ok());
    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{}, {-2, 4}, {-3, 4}}));

    llvm::SmallVector<int64_t> values = {-2, 4, -3, 4};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }

  {  // Both dynamic dimensions are the same.
    Arguments<OpaqueArg, MemrefDesc> arguments(3);
    arguments.push_back(GetFakeOpaqueArg());
    arguments.push_back(GetFakeMemref({2, 4}));
    arguments.push_back(GetFakeMemref({2, 4}));

    auto symbolic = resolver.Resolve(arguments);
    auto hash = resolver.ResolveHash(arguments);

    ASSERT_TRUE(symbolic.ok());
    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{}, {-2, 4}, {-2, 4}}));

    llvm::SmallVector<int64_t> values = {-2, 4, -2, 4};
    EXPECT_EQ(*hash, llvm::hash_combine_range(values.begin(), values.end()));
  }
}

// -------------------------------------------------------------------------- //
// Performance benchmarks are below.
// -------------------------------------------------------------------------- //

struct Resolve {
  static absl::StatusOr<llvm::SmallVector<SymbolicShape>> Run(
      SymbolicShapesResolver& resolver, ArrayRef<MemrefDesc> operands) {
    return resolver.Resolve(operands);
  }
};

struct ResolveHash {
  static absl::StatusOr<llvm::hash_code> Run(SymbolicShapesResolver& resolver,
                                             ArrayRef<MemrefDesc> operands) {
    return resolver.ResolveHash(operands);
  }
};

template <typename Resolver>
static void BenchmarkFullyDynamic(benchmark::State& state) {
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::PRED,
                 PrimitiveType::F32};

  auto type =
      GetFunctionType(dtypes, {{{MemrefType::kDynamic, MemrefType::kDynamic}},
                               {{MemrefType::kDynamic, MemrefType::kDynamic}},
                               {{MemrefType::kDynamic, MemrefType::kDynamic}},
                               {{MemrefType::kDynamic, MemrefType::kDynamic}}});

  auto constraints = {
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved,
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  auto operands = GetFakeMemrefs({{1, 2}, {3, 4}, {5, 6}, {7, 8}});

  for (auto _ : state) {
    auto result = Resolver::Run(resolver, operands);
    benchmark::DoNotOptimize(*result);
  }
}

template <typename Resolver>
static void BenchmarkSameDynamic(benchmark::State& state) {
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::PRED,
                 PrimitiveType::F32};

  auto type =
      GetFunctionType(dtypes, {{{MemrefType::kDynamic, MemrefType::kDynamic}},
                               {{MemrefType::kDynamic, MemrefType::kDynamic}},
                               {{MemrefType::kDynamic, MemrefType::kDynamic}},
                               {{MemrefType::kDynamic, MemrefType::kDynamic}}});

  auto constraints = {
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved,
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  auto operands = GetFakeMemrefs({{2, 2}, {2, 2}, {2, 2}, {2, 2}});

  for (auto _ : state) {
    auto result = Resolver::Run(resolver, operands);
    benchmark::DoNotOptimize(*result);
  }
}

template <typename Resolver>
static void BenchmarkSomeDynamic(benchmark::State& state) {
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::PRED,
                 PrimitiveType::F32};

  auto type =
      GetFunctionType(dtypes, {{{2, 2}},
                               {{4, 4}},
                               {{8, 8}},
                               {{MemrefType::kDynamic, MemrefType::kDynamic}}});

  auto constraints = {
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved,
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  auto operands = GetFakeMemrefs({{2, 2}, {4, 4}, {8, 8}, {16, 16}});

  for (auto _ : state) {
    auto result = Resolver::Run(resolver, operands);
    benchmark::DoNotOptimize(*result);
  }
}

template <typename Resolver>
static void BenchmarkStatic(benchmark::State& state) {
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::PRED,
                 PrimitiveType::F32};

  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamic, 4}},
                                       {{MemrefType::kDynamic, 8}},
                                       {{MemrefType::kDynamic, 16}},
                                       {{MemrefType::kDynamic, 32}}});

  auto constraints = {
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved,
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  auto operands = GetFakeMemrefs({{32, 4}, {16, 8}, {8, 16}, {4, 32}});

  for (auto _ : state) {
    auto result = Resolver::Run(resolver, operands);
    benchmark::DoNotOptimize(*result);
  }
}

template <typename Resolver>
static void BenchmarkSymbolic(benchmark::State& state) {
  auto dtypes = {PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::PRED,
                 PrimitiveType::F32};

  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamic, 4}},
                                       {{MemrefType::kDynamic, 8}},
                                       {{MemrefType::kDynamic, 16}},
                                       {{MemrefType::kDynamic, 32}}});

  auto constraints = {
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved,
      ArgumentConstraint::kResolved, ArgumentConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  auto operands = GetFakeMemrefs({{1, 4}, {2, 8}, {3, 16}, {4, 32}});

  for (auto _ : state) {
    auto result = Resolver::Run(resolver, operands);
    benchmark::DoNotOptimize(*result);
  }
}

// -------------------------------------------------------------------------- //
// Run benchmarks for resolving symbolic shapes.
// -------------------------------------------------------------------------- //

static void BM_ResolveFullyDynamic(benchmark::State& state) {
  BenchmarkFullyDynamic<Resolve>(state);
}

static void BM_ResolveSameDynamic(benchmark::State& state) {
  BenchmarkSameDynamic<Resolve>(state);
}

static void BM_ResolveSomeDynamic(benchmark::State& state) {
  BenchmarkSomeDynamic<Resolve>(state);
}

static void BM_ResolveAsStatic(benchmark::State& state) {
  BenchmarkStatic<Resolve>(state);
}

static void BM_ResolveAsSymbolic(benchmark::State& state) {
  BenchmarkSymbolic<Resolve>(state);
}

BENCHMARK(BM_ResolveFullyDynamic);
BENCHMARK(BM_ResolveSameDynamic);
BENCHMARK(BM_ResolveSomeDynamic);
BENCHMARK(BM_ResolveAsStatic);
BENCHMARK(BM_ResolveAsSymbolic);

// -------------------------------------------------------------------------- //
// Run benchmarks for resolving and computing a hash of symbolic shapes.
// -------------------------------------------------------------------------- //

static void BM_ResolveHashFullyDynamic(benchmark::State& state) {
  BenchmarkFullyDynamic<ResolveHash>(state);
}

static void BM_ResolveHashSameDynamic(benchmark::State& state) {
  BenchmarkSameDynamic<ResolveHash>(state);
}

static void BM_ResolveHashSomeDynamic(benchmark::State& state) {
  BenchmarkSomeDynamic<ResolveHash>(state);
}

static void BM_ResolveHashAsStatic(benchmark::State& state) {
  BenchmarkStatic<ResolveHash>(state);
}

static void BM_ResolveHashAsSymbolic(benchmark::State& state) {
  BenchmarkSymbolic<ResolveHash>(state);
}

BENCHMARK(BM_ResolveHashFullyDynamic);
BENCHMARK(BM_ResolveHashSameDynamic);
BENCHMARK(BM_ResolveHashSomeDynamic);
BENCHMARK(BM_ResolveHashAsStatic);
BENCHMARK(BM_ResolveHashAsSymbolic);

// -------------------------------------------------------------------------- //
// Run benchmarks for hashing resolved symbolic shapes.
// -------------------------------------------------------------------------- //

static void HashSymbolicShapes(benchmark::State& state,
                               ArrayRef<SymbolicShape> symbolic_shapes) {
  for (auto _ : state) {
    auto hash = SymbolicShapesResolver::Hash(symbolic_shapes);
    benchmark::DoNotOptimize(hash);
  }
}

static void BM_Hash1x1(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(1, {1});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash1x4(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(4, {1});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash1x8(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(8, {1});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash2x1(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(1, {1, 2});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash2x4(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(4, {1, 2});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash2x8(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(8, {1, 2});
  HashSymbolicShapes(state, symbolic_shapes);
}

BENCHMARK(BM_Hash1x1);
BENCHMARK(BM_Hash1x4);
BENCHMARK(BM_Hash1x8);
BENCHMARK(BM_Hash2x1);
BENCHMARK(BM_Hash2x4);
BENCHMARK(BM_Hash2x8);

}  // namespace runtime
}  // namespace xla
