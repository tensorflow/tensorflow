/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/c/c_api_opaque.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api.h"

namespace tflite {
namespace {

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithMemNoneBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithMmapRoBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithArenaRwBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithDynamicBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithPersistentRoBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithCustomBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithVariantObjectBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithMemNoneBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithMmapRoBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithArenaRwBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithDynamicBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithPersistentRoBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithCustomBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithVariantObjectBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithMemNoneBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithMmapRoBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithArenaRwBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithDynamicBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithPersistentRoBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithCustomBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithVariantObjectBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithMemNoneBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithMmapRoBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithArenaRwBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithDynamicBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithPersistentRoBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithCustomBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithVariantObjectBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithMemNoneBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithMmapRoBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithArenaRwBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithDynamicBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithPersistentRoBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithCustomBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithVariantObjectBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

}  // namespace
}  // namespace tflite
