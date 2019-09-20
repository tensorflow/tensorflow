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

#[macro_use]
extern crate bencher;
use bencher::Bencher;

extern crate flatbuffers;

#[allow(dead_code, unused_imports)]
#[path = "../../monster_test_generated.rs"]
mod monster_test_generated;
pub use monster_test_generated::my_game;

fn traverse_canonical_buffer(bench: &mut Bencher) {
    let owned_data = {
        let mut builder = &mut flatbuffers::FlatBufferBuilder::new();
        create_serialized_example_with_generated_code(&mut builder, true);
        builder.finished_data().to_vec()
    };
    let data = &owned_data[..];
    let n = data.len() as u64;
    bench.iter(|| {
        traverse_serialized_example_with_generated_code(data);
    });
    bench.bytes = n;
}

fn create_canonical_buffer_then_reset(bench: &mut Bencher) {
    let mut builder = &mut flatbuffers::FlatBufferBuilder::new();
    // warmup
    create_serialized_example_with_generated_code(&mut builder, true);
    let n = builder.finished_data().len() as u64;
    builder.reset();

    bench.iter(|| {
        let _ = create_serialized_example_with_generated_code(&mut builder, true);
        builder.reset();
    });

    bench.bytes = n;
}

#[inline(always)]
fn create_serialized_example_with_generated_code(builder: &mut flatbuffers::FlatBufferBuilder, finish: bool) -> usize{
    let s0 = builder.create_string("test1");
    let s1 = builder.create_string("test2");
    let t0_name = builder.create_string("Barney");
    let t1_name = builder.create_string("Fred");
    let t2_name = builder.create_string("Wilma");
    let t0 = my_game::example::Monster::create(builder, &my_game::example::MonsterArgs{
        hp: 1000,
        name: Some(t0_name),
        ..Default::default()
    });
    let t1 = my_game::example::Monster::create(builder, &my_game::example::MonsterArgs{
        name: Some(t1_name),
        ..Default::default()
    });
    let t2 = my_game::example::Monster::create(builder, &my_game::example::MonsterArgs{
        name: Some(t2_name),
        ..Default::default()
    });
    let mon = {
        let name = builder.create_string("MyMonster");
        let fred_name = builder.create_string("Fred");
        let inventory = builder.create_vector_direct(&[0u8, 1, 2, 3, 4]);
        let test4 = builder.create_vector_direct(&[my_game::example::Test::new(10, 20),
                                                   my_game::example::Test::new(30, 40)]);
        let pos = my_game::example::Vec3::new(1.0, 2.0, 3.0, 3.0, my_game::example::Color::Green, &my_game::example::Test::new(5i16, 6i8));
        let args = my_game::example::MonsterArgs{
            hp: 80,
            mana: 150,
            name: Some(name),
            pos: Some(&pos),
            test_type: my_game::example::Any::Monster,
            test: Some(my_game::example::Monster::create(builder, &my_game::example::MonsterArgs{
                name: Some(fred_name),
                ..Default::default()
            }).as_union_value()),
            inventory: Some(inventory),
            test4: Some(test4),
            testarrayofstring: Some(builder.create_vector(&[s0, s1])),
            testarrayoftables: Some(builder.create_vector(&[t0, t1, t2])),
            ..Default::default()
        };
        my_game::example::Monster::create(builder, &args)
    };
    if finish {
        my_game::example::finish_monster_buffer(builder, mon);
    }

    builder.finished_data().len()

    // make it do some work
    // if builder.finished_data().len() == 0 { panic!("bad benchmark"); }
}

#[inline(always)]
fn blackbox<T>(t: T) -> T {
    // encapsulate this in case we need to turn it into a noop
    bencher::black_box(t)
}

#[inline(always)]
fn traverse_serialized_example_with_generated_code(bytes: &[u8]) {
    let m = my_game::example::get_root_as_monster(bytes);
    blackbox(m.hp());
    blackbox(m.mana());
    blackbox(m.name());
    let pos = m.pos().unwrap();
    blackbox(pos.x());
    blackbox(pos.y());
    blackbox(pos.z());
    blackbox(pos.test1());
    blackbox(pos.test2());
    let pos_test3 = pos.test3();
    blackbox(pos_test3.a());
    blackbox(pos_test3.b());
    blackbox(m.test_type());
    let table2 = m.test().unwrap();
    let monster2 = my_game::example::Monster::init_from_table(table2);
    blackbox(monster2.name());
    blackbox(m.inventory());
    blackbox(m.test4());
    let testarrayoftables = m.testarrayoftables().unwrap();
    blackbox(testarrayoftables.get(0).hp());
    blackbox(testarrayoftables.get(0).name());
    blackbox(testarrayoftables.get(1).name());
    blackbox(testarrayoftables.get(2).name());
    let testarrayofstring = m.testarrayofstring().unwrap();
    blackbox(testarrayofstring.get(0));
    blackbox(testarrayofstring.get(1));
}

fn create_string_10(bench: &mut Bencher) {
    let builder = &mut flatbuffers::FlatBufferBuilder::new_with_capacity(1<<20);
    let mut i = 0;
    bench.iter(|| {
        builder.create_string("foobarbaz"); // zero-terminated -> 10 bytes
        i += 1;
        if i == 10000 {
            builder.reset();
            i = 0;
        }
    });

    bench.bytes = 10;
}

fn create_string_100(bench: &mut Bencher) {
    let builder = &mut flatbuffers::FlatBufferBuilder::new_with_capacity(1<<20);
    let s_owned = (0..99).map(|_| "x").collect::<String>();
    let s: &str = &s_owned;

    let mut i = 0;
    bench.iter(|| {
        builder.create_string(s); // zero-terminated -> 100 bytes
        i += 1;
        if i == 1000 {
            builder.reset();
            i = 0;
        }
    });

    bench.bytes = s.len() as u64;
}

fn create_byte_vector_100_naive(bench: &mut Bencher) {
    let builder = &mut flatbuffers::FlatBufferBuilder::new_with_capacity(1<<20);
    let v_owned = (0u8..100).map(|i| i).collect::<Vec<u8>>();
    let v: &[u8] = &v_owned;

    let mut i = 0;
    bench.iter(|| {
        builder.create_vector(v); // zero-terminated -> 100 bytes
        i += 1;
        if i == 10000 {
            builder.reset();
            i = 0;
        }
    });

    bench.bytes = v.len() as u64;
}

fn create_byte_vector_100_optimal(bench: &mut Bencher) {
    let builder = &mut flatbuffers::FlatBufferBuilder::new_with_capacity(1<<20);
    let v_owned = (0u8..100).map(|i| i).collect::<Vec<u8>>();
    let v: &[u8] = &v_owned;

    let mut i = 0;
    bench.iter(|| {
        builder.create_vector_direct(v);
        i += 1;
        if i == 10000 {
            builder.reset();
            i = 0;
        }
    });

    bench.bytes = v.len() as u64;
}

benchmark_group!(benches, create_byte_vector_100_naive, create_byte_vector_100_optimal, traverse_canonical_buffer, create_canonical_buffer_then_reset, create_string_10, create_string_100);
benchmark_main!(benches);
