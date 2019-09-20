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

use std::mem::size_of;

/// Trait for values that must be stored in little-endian byte order, but
/// might be represented in memory as big-endian. Every type that implements
/// EndianScalar is a valid FlatBuffers scalar value.
///
/// The Rust stdlib does not provide a trait to represent scalars, so this trait
/// serves that purpose, too.
///
/// Note that we do not use the num-traits crate for this, because it provides
/// "too much". For example, num-traits provides i128 support, but that is an
/// invalid FlatBuffers type.
pub trait EndianScalar: Sized + PartialEq + Copy + Clone {
    fn to_little_endian(self) -> Self;
    fn from_little_endian(self) -> Self;
}

/// Macro for implementing a no-op endian conversion. This is used for types
/// that are one byte wide.
macro_rules! impl_endian_scalar_noop {
    ($ty:ident) => {
        impl EndianScalar for $ty {
            #[inline]
            fn to_little_endian(self) -> Self {
                self
            }
            #[inline]
            fn from_little_endian(self) -> Self {
                self
            }
        }
    };
}

/// Macro for implementing an endian conversion using the stdlib `to_le` and
/// `from_le` functions. This is used for integer types. It is not used for
/// floats, because the `to_le` and `from_le` are not implemented for them in
/// the stdlib.
macro_rules! impl_endian_scalar_stdlib_le_conversion {
    ($ty:ident) => {
        impl EndianScalar for $ty {
            #[inline]
            fn to_little_endian(self) -> Self {
                Self::to_le(self)
            }
            #[inline]
            fn from_little_endian(self) -> Self {
                Self::from_le(self)
            }
        }
    };
}

impl_endian_scalar_noop!(bool);
impl_endian_scalar_noop!(u8);
impl_endian_scalar_noop!(i8);

impl_endian_scalar_stdlib_le_conversion!(u16);
impl_endian_scalar_stdlib_le_conversion!(u32);
impl_endian_scalar_stdlib_le_conversion!(u64);
impl_endian_scalar_stdlib_le_conversion!(i16);
impl_endian_scalar_stdlib_le_conversion!(i32);
impl_endian_scalar_stdlib_le_conversion!(i64);

impl EndianScalar for f32 {
    /// Convert f32 from host endian-ness to little-endian.
    #[inline]
    fn to_little_endian(self) -> Self {
        #[cfg(target_endian = "little")]
        {
            self
        }
        #[cfg(not(target_endian = "little"))]
        {
            byte_swap_f32(self)
        }
    }
    /// Convert f32 from little-endian to host endian-ness.
    #[inline]
    fn from_little_endian(self) -> Self {
        #[cfg(target_endian = "little")]
        {
            self
        }
        #[cfg(not(target_endian = "little"))]
        {
            byte_swap_f32(self)
        }
    }
}

impl EndianScalar for f64 {
    /// Convert f64 from host endian-ness to little-endian.
    #[inline]
    fn to_little_endian(self) -> Self {
        #[cfg(target_endian = "little")]
        {
            self
        }
        #[cfg(not(target_endian = "little"))]
        {
            byte_swap_f64(self)
        }
    }
    /// Convert f64 from little-endian to host endian-ness.
    #[inline]
    fn from_little_endian(self) -> Self {
        #[cfg(target_endian = "little")]
        {
            self
        }
        #[cfg(not(target_endian = "little"))]
        {
            byte_swap_f64(self)
        }
    }
}

/// Swaps the bytes of an f32.
#[allow(dead_code)]
#[inline]
pub fn byte_swap_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits().swap_bytes())
}

/// Swaps the bytes of an f64.
#[allow(dead_code)]
#[inline]
pub fn byte_swap_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits().swap_bytes())
}

/// Place an EndianScalar into the provided mutable byte slice. Performs
/// endian conversion, if necessary.
#[inline]
pub fn emplace_scalar<T: EndianScalar>(s: &mut [u8], x: T) {
    let sz = size_of::<T>();
    let mut_ptr = (&mut s[..sz]).as_mut_ptr() as *mut T;
    let val = x.to_little_endian();
    unsafe {
        *mut_ptr = val;
    }
}

/// Read an EndianScalar from the provided byte slice at the specified location.
/// Performs endian conversion, if necessary.
#[inline]
pub fn read_scalar_at<T: EndianScalar>(s: &[u8], loc: usize) -> T {
    let buf = &s[loc..loc + size_of::<T>()];
    read_scalar(buf)
}

/// Read an EndianScalar from the provided byte slice. Performs endian
/// conversion, if necessary.
#[inline]
pub fn read_scalar<T: EndianScalar>(s: &[u8]) -> T {
    let sz = size_of::<T>();

    let p = (&s[..sz]).as_ptr() as *const T;
    let x = unsafe { *p };

    x.from_little_endian()
}
