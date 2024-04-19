/*
 * Copyright 2022 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xla/mlir/runtime/transforms/type_converter.h"

#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/runtime/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace runtime {

using mlir::FloatType;
using mlir::IntegerType;
using mlir::MLIRContext;

TEST(TypeConverterTest, ScalarTypeConversion) {
  MLIRContext ctx;
  auto i32 = IntegerType::get(&ctx, 32);
  auto f32 = FloatType::getF32(&ctx);

  TypeConverter converter;
  auto i32_converted = converter.Convert(i32);
  auto f32_converted = converter.Convert(f32);

  EXPECT_TRUE(i32_converted.ok() && *i32_converted);
  EXPECT_TRUE(f32_converted.ok() && *f32_converted);

  auto* i32_scalar = llvm::dyn_cast<ScalarType>(i32_converted->get());
  EXPECT_TRUE(i32_scalar);
  ASSERT_EQ(i32_scalar->type(), PrimitiveType::S32);

  auto* f32_scalar = llvm::dyn_cast<ScalarType>(f32_converted->get());
  EXPECT_TRUE(f32_scalar);
  ASSERT_EQ(f32_scalar->type(), PrimitiveType::F32);
}

}  // namespace runtime
}  // namespace xla
