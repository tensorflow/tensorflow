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

use std::marker::PhantomData;
use std::mem::size_of;
use std::slice::from_raw_parts;
use std::str::from_utf8_unchecked;

#[cfg(target_endian = "little")]
use endian_scalar::EndianScalar;
use endian_scalar::{read_scalar, read_scalar_at};
use follow::Follow;
use primitives::*;

#[derive(Debug, Clone, Copy)]
pub struct Vector<'a, T: 'a>(&'a [u8], usize, PhantomData<T>);

impl<'a, T: 'a> Vector<'a, T> {
    #[inline(always)]
    pub fn new(buf: &'a [u8], loc: usize) -> Self {
        Vector {
            0: buf,
            1: loc,
            2: PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        read_scalar::<UOffsetT>(&self.0[self.1 as usize..]) as usize
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T: Follow<'a> + 'a> Vector<'a, T> {
    #[inline(always)]
    pub fn get(&self, idx: usize) -> T::Inner {
        debug_assert!(idx < read_scalar::<u32>(&self.0[self.1 as usize..]) as usize);
        let sz = size_of::<T>();
        debug_assert!(sz > 0);
        T::follow(self.0, self.1 as usize + SIZE_UOFFSET + sz * idx)
    }
}

pub trait SafeSliceAccess {}
impl<'a, T: SafeSliceAccess + 'a> Vector<'a, T> {
    pub fn safe_slice(self) -> &'a [T] {
        let buf = self.0;
        let loc = self.1;
        let sz = size_of::<T>();
        debug_assert!(sz > 0);
        let len = read_scalar_at::<UOffsetT>(&buf, loc) as usize;
        let data_buf = &buf[loc + SIZE_UOFFSET..loc + SIZE_UOFFSET + len * sz];
        let ptr = data_buf.as_ptr() as *const T;
        let s: &'a [T] = unsafe { from_raw_parts(ptr, len) };
        s
    }
}

impl SafeSliceAccess for u8 {}
impl SafeSliceAccess for i8 {}
impl SafeSliceAccess for bool {}

#[cfg(target_endian = "little")]
mod le_safe_slice_impls {
    impl super::SafeSliceAccess for u16 {}
    impl super::SafeSliceAccess for u32 {}
    impl super::SafeSliceAccess for u64 {}

    impl super::SafeSliceAccess for i16 {}
    impl super::SafeSliceAccess for i32 {}
    impl super::SafeSliceAccess for i64 {}

    impl super::SafeSliceAccess for f32 {}
    impl super::SafeSliceAccess for f64 {}
}

#[cfg(target_endian = "little")]
pub use self::le_safe_slice_impls::*;

pub fn follow_cast_ref<'a, T: Sized + 'a>(buf: &'a [u8], loc: usize) -> &'a T {
    let sz = size_of::<T>();
    let buf = &buf[loc..loc + sz];
    let ptr = buf.as_ptr() as *const T;
    unsafe { &*ptr }
}

impl<'a> Follow<'a> for &'a str {
    type Inner = &'a str;
    fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
        let len = read_scalar_at::<UOffsetT>(&buf, loc) as usize;
        let slice = &buf[loc + SIZE_UOFFSET..loc + SIZE_UOFFSET + len];
        unsafe { from_utf8_unchecked(slice) }
    }
}

#[cfg(target_endian = "little")]
fn follow_slice_helper<T>(buf: &[u8], loc: usize) -> &[T] {
    let sz = size_of::<T>();
    debug_assert!(sz > 0);
    let len = read_scalar_at::<UOffsetT>(&buf, loc) as usize;
    let data_buf = &buf[loc + SIZE_UOFFSET..loc + SIZE_UOFFSET + len * sz];
    let ptr = data_buf.as_ptr() as *const T;
    let s: &[T] = unsafe { from_raw_parts(ptr, len) };
    s
}

/// Implement direct slice access if the host is little-endian.
#[cfg(target_endian = "little")]
impl<'a, T: EndianScalar> Follow<'a> for &'a [T] {
    type Inner = &'a [T];
    fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
        follow_slice_helper::<T>(buf, loc)
    }
}

/// Implement Follow for all possible Vectors that have Follow-able elements.
impl<'a, T: Follow<'a> + 'a> Follow<'a> for Vector<'a, T> {
    type Inner = Vector<'a, T>;
    fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
        Vector::new(buf, loc)
    }
}
