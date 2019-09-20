// define a passthrough allocator that tracks alloc calls.
// (note that we can't drop this in to the usual test suite, because it's a big
// global variable).
use std::alloc::{GlobalAlloc, Layout, System};

static mut N_ALLOCS: usize = 0;

struct TrackingAllocator;

impl TrackingAllocator {
    fn n_allocs(&self) -> usize {
        unsafe { N_ALLOCS }
    }
}
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        N_ALLOCS += 1;
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

// use the tracking allocator:
#[global_allocator]
static A: TrackingAllocator = TrackingAllocator;

// import the flatbuffers generated code:
extern crate flatbuffers;
#[allow(dead_code, unused_imports)]
#[path = "../../monster_test_generated.rs"]
mod monster_test_generated;
pub use monster_test_generated::my_game;

// verbatim from the test suite:
fn create_serialized_example_with_generated_code(builder: &mut flatbuffers::FlatBufferBuilder) {
    let mon = {
        let _ = builder.create_vector_of_strings(&["these", "unused", "strings", "check", "the", "create_vector_of_strings", "function"]);

        let s0 = builder.create_string("test1");
        let s1 = builder.create_string("test2");
        let fred_name = builder.create_string("Fred");

        // can't inline creation of this Vec3 because we refer to it by reference, so it must live
        // long enough to be used by MonsterArgs.
        let pos = my_game::example::Vec3::new(1.0, 2.0, 3.0, 3.0, my_game::example::Color::Green, &my_game::example::Test::new(5i16, 6i8));

        let args = my_game::example::MonsterArgs{
            hp: 80,
            mana: 150,
            name: Some(builder.create_string("MyMonster")),
            pos: Some(&pos),
            test_type: my_game::example::Any::Monster,
            test: Some(my_game::example::Monster::create(builder, &my_game::example::MonsterArgs{
                name: Some(fred_name),
                ..Default::default()
            }).as_union_value()),
            inventory: Some(builder.create_vector_direct(&[0u8, 1, 2, 3, 4][..])),
            test4: Some(builder.create_vector_direct(&[my_game::example::Test::new(10, 20),
            my_game::example::Test::new(30, 40)])),
            testarrayofstring: Some(builder.create_vector(&[s0, s1])),
            ..Default::default()
        };
        my_game::example::Monster::create(builder, &args)
    };
    my_game::example::finish_monster_buffer(builder, mon);
}

fn main() {
    // test the allocation tracking:
    {
        let before = A.n_allocs();
        let _x: Vec<u8> = vec![0u8; 1];
        let after = A.n_allocs();
        assert_eq!(before + 1, after);
    }

    let builder = &mut flatbuffers::FlatBufferBuilder::new();
    {
        // warm up the builder (it can make small allocs internally, such as for storing vtables):
        create_serialized_example_with_generated_code(builder);
    }

    // reset the builder, clearing its heap-allocated memory:
    builder.reset();

    {
        let before = A.n_allocs();
        create_serialized_example_with_generated_code(builder);
        let after = A.n_allocs();
        assert_eq!(before, after, "KO: Heap allocs occurred in Rust write path");
    }

    let buf = builder.finished_data();

    // use the allocation tracking on the read path:
    {
        let before = A.n_allocs();

        // do many reads, forcing them to execute by using assert_eq:
        {
            let m = my_game::example::get_root_as_monster(buf);
            assert_eq!(80, m.hp());
            assert_eq!(150, m.mana());
            assert_eq!("MyMonster", m.name());

            let pos = m.pos().unwrap();
            // We know the bits should be exactly equal here but compilers may
            // optimize floats in subtle ways so we're playing it safe and using
            // epsilon comparison
            assert!((pos.x() - 1.0f32).abs() < std::f32::EPSILON);
            assert!((pos.y() - 2.0f32).abs() < std::f32::EPSILON);
            assert!((pos.z() - 3.0f32).abs() < std::f32::EPSILON);
            assert!((pos.test1() - 3.0f64).abs() < std::f64::EPSILON);
            assert_eq!(pos.test2(), my_game::example::Color::Green);
            let pos_test3 = pos.test3();
            assert_eq!(pos_test3.a(), 5i16);
            assert_eq!(pos_test3.b(), 6i8);
            assert_eq!(m.test_type(), my_game::example::Any::Monster);
            let table2 = m.test().unwrap();
            let m2 = my_game::example::Monster::init_from_table(table2);

            assert_eq!(m2.name(), "Fred");

            let inv = m.inventory().unwrap();
            assert_eq!(inv.len(), 5);
            assert_eq!(inv.iter().sum::<u8>(), 10u8);

            let test4 = m.test4().unwrap();
            assert_eq!(test4.len(), 2);
            assert_eq!(i32::from(test4[0].a()) + i32::from(test4[1].a()) +
                       i32::from(test4[0].b()) + i32::from(test4[1].b()), 100);

            let testarrayofstring = m.testarrayofstring().unwrap();
            assert_eq!(testarrayofstring.len(), 2);
            assert_eq!(testarrayofstring.get(0), "test1");
            assert_eq!(testarrayofstring.get(1), "test2");
        }

        // assert that no allocs occurred:
        let after = A.n_allocs();
        assert_eq!(before, after, "KO: Heap allocs occurred in Rust read path");
    }
    println!("Rust: Heap alloc checks completed successfully");
}
