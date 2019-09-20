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

use endian_scalar::read_scalar_at;
use follow::Follow;
use primitives::*;

/// VTable encapsulates read-only usage of a vtable. It is only to be used
/// by generated code.
#[derive(Debug)]
pub struct VTable<'a> {
    buf: &'a [u8],
    loc: usize,
}

impl<'a> PartialEq for VTable<'a> {
    fn eq(&self, other: &VTable) -> bool {
        self.as_bytes().eq(other.as_bytes())
    }
}

impl<'a> VTable<'a> {
    pub fn init(buf: &'a [u8], loc: usize) -> Self {
        VTable { buf, loc }
    }
    pub fn num_fields(&self) -> usize {
        (self.num_bytes() / SIZE_VOFFSET) - 2
    }
    pub fn num_bytes(&self) -> usize {
        read_scalar_at::<VOffsetT>(self.buf, self.loc) as usize
    }
    pub fn object_inline_num_bytes(&self) -> usize {
        let n = read_scalar_at::<VOffsetT>(self.buf, self.loc + SIZE_VOFFSET);
        n as usize
    }
    pub fn get_field(&self, idx: usize) -> VOffsetT {
        // TODO(rw): distinguish between None and 0?
        if idx > self.num_fields() {
            return 0;
        }
        read_scalar_at::<VOffsetT>(
            self.buf,
            self.loc + SIZE_VOFFSET + SIZE_VOFFSET + SIZE_VOFFSET * idx,
        )
    }
    pub fn get(&self, byte_loc: VOffsetT) -> VOffsetT {
        // TODO(rw): distinguish between None and 0?
        if byte_loc as usize >= self.num_bytes() {
            return 0;
        }
        read_scalar_at::<VOffsetT>(self.buf, self.loc + byte_loc as usize)
    }
    pub fn as_bytes(&self) -> &[u8] {
        let len = self.num_bytes();
        &self.buf[self.loc..self.loc + len]
    }
}

#[allow(dead_code)]
pub fn field_index_to_field_offset(field_id: VOffsetT) -> VOffsetT {
    // Should correspond to what end_table() below builds up.
    let fixed_fields = 2; // Vtable size and Object Size.
    ((field_id + fixed_fields) * (SIZE_VOFFSET as VOffsetT)) as VOffsetT
}

#[allow(dead_code)]
pub fn field_offset_to_field_index(field_o: VOffsetT) -> VOffsetT {
    debug_assert!(field_o >= 2);
    let fixed_fields = 2; // VTable size and Object Size.
    (field_o / (SIZE_VOFFSET as VOffsetT)) - fixed_fields
}

impl<'a> Follow<'a> for VTable<'a> {
    type Inner = VTable<'a>;
    fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
        VTable::init(buf, loc)
    }
}
