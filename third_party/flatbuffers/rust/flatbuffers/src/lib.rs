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

//! # FlatBuffers
//!
//! A library for memory-efficient serialization of data.
//!
//! This crate provides runtime support for the FlatBuffers format in the Rust programming language.
//! To use this crate, first generate code with the `flatc` compiler, as described here: https://google.github.io/flatbuffers/
//! Then, include that code into your project.
//! Finally, add this crate to your `Cargo.toml`.
//!
//! At this time, Rust support is experimental, and APIs may change between minor versions.
//!
//! At this time, to generate Rust code, you will need the latest `master` version of `flatc`, available from here: https://github.com/google/flatbuffers
//! (On OSX, you can install FlatBuffers from `HEAD` with the Homebrew package manager.)

mod builder;
mod endian_scalar;
mod follow;
mod primitives;
mod push;
mod table;
mod vector;
mod vtable;
mod vtable_writer;

pub use builder::FlatBufferBuilder;
pub use endian_scalar::{
    byte_swap_f32, byte_swap_f64, emplace_scalar, read_scalar, read_scalar_at, EndianScalar,
};
pub use follow::{Follow, FollowStart};
pub use primitives::*;
pub use push::Push;
pub use table::{buffer_has_identifier, get_root, get_size_prefixed_root, Table};
pub use vector::{follow_cast_ref, SafeSliceAccess, Vector};
pub use vtable::field_index_to_field_offset;

// TODO(rw): Unify `create_vector` and `create_vector_direct` by using
//           `Into<Vector<...>>`.
// TODO(rw): Split fill ops in builder into fill_small, fill_big like in C++.
