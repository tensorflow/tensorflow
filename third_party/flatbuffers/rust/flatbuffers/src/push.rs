/*
 * Copyright 2018 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::cmp::max;
use std::mem::{align_of, size_of};

use endian_scalar::emplace_scalar;

/// Trait to abstract over functionality needed to write values (either owned
/// or referenced). Used in FlatBufferBuilder and implemented for generated
/// types.
pub trait Push: Sized {
    type Output;
    fn push(&self, dst: &mut [u8], _rest: &[u8]);
    #[inline]
    fn size() -> usize {
        size_of::<Self::Output>()
    }
    #[inline]
    fn alignment() -> PushAlignment {
        PushAlignment::new(align_of::<Self::Output>())
    }
}

/// Ensure Push alignment calculations are typesafe (because this helps reduce
/// implementation issues when using FlatBufferBuilder::align).
pub struct PushAlignment(usize);
impl PushAlignment {
    #[inline]
    pub fn new(x: usize) -> Self {
        PushAlignment { 0: x }
    }
    #[inline]
    pub fn value(&self) -> usize {
        self.0
    }
    #[inline]
    pub fn max_of(&self, o: usize) -> Self {
        PushAlignment::new(max(self.0, o))
    }
}

/// Macro to implement Push for EndianScalar types.
macro_rules! impl_push_for_endian_scalar {
    ($ty:ident) => {
        impl Push for $ty {
            type Output = $ty;

            #[inline]
            fn push(&self, dst: &mut [u8], _rest: &[u8]) {
                emplace_scalar::<$ty>(dst, *self);
            }
        }
    };
}

impl_push_for_endian_scalar!(bool);
impl_push_for_endian_scalar!(u8);
impl_push_for_endian_scalar!(i8);
impl_push_for_endian_scalar!(u16);
impl_push_for_endian_scalar!(i16);
impl_push_for_endian_scalar!(u32);
impl_push_for_endian_scalar!(i32);
impl_push_for_endian_scalar!(u64);
impl_push_for_endian_scalar!(i64);
impl_push_for_endian_scalar!(f32);
impl_push_for_endian_scalar!(f64);
