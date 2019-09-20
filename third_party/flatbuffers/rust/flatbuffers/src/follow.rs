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

/// Follow is a trait that allows us to access FlatBuffers in a declarative,
/// type safe, and fast way. They compile down to almost no code (after
/// optimizations). Conceptually, Follow lifts the offset-based access
/// patterns of FlatBuffers data into the type system. This trait is used
/// pervasively at read time, to access tables, vtables, vectors, strings, and
/// all other data. At this time, Follow is not utilized much on the write
/// path.
///
/// Writing a new Follow implementation primarily involves deciding whether
/// you want to return data (of the type Self::Inner) or do you want to
/// continue traversing the FlatBuffer.
pub trait Follow<'a> {
    type Inner;
    fn follow(buf: &'a [u8], loc: usize) -> Self::Inner;
}

/// Execute a follow as a top-level function.
#[allow(dead_code)]
#[inline]
pub fn lifted_follow<'a, T: Follow<'a>>(buf: &'a [u8], loc: usize) -> T::Inner {
    T::follow(buf, loc)
}

/// FollowStart wraps a Follow impl in a struct type. This can make certain
/// programming patterns more ergonomic.
#[derive(Debug)]
pub struct FollowStart<T>(PhantomData<T>);
impl<'a, T: Follow<'a> + 'a> FollowStart<T> {
    #[inline]
    pub fn new() -> Self {
        Self { 0: PhantomData }
    }
    #[inline]
    pub fn self_follow(&'a self, buf: &'a [u8], loc: usize) -> T::Inner {
        T::follow(buf, loc)
    }
}
impl<'a, T: Follow<'a>> Follow<'a> for FollowStart<T> {
    type Inner = T::Inner;
    #[inline]
    fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
        T::follow(buf, loc)
    }
}
