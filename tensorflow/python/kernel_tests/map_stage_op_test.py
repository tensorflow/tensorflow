import numpy as np

import tensorflow as tf

from map_staging_area import MapStagingArea

blah_key = tf.string_to_hash_bucket_fast('blah', 2**32)
foo_key = tf.string_to_hash_bucket_fast('foo', 2**32)

dtypes = [tf.float32, tf.float32, tf.int32]
capacity = 2
msa = MapStagingArea(dtypes, capacity=capacity, ordered=True)

with tf.device('/gpu:0'):
    data1 = [tf.constant(d, dtype=dt) for d, dt in zip([1.0, 3.0, 4], dtypes)]
    data2 = [tf.constant(d, dtype=dt) for d, dt in zip([2.0, 5.0, 6], dtypes)]

    stage1 = msa.put(blah_key, data1) if True else tf.no_op()
    stage2 = msa.put(foo_key, data2) if True else tf.no_op()
    get = msa.get(blah_key) if True else tf.no_op()
    unstage = msa.pop(blah_key) if True else tf.no_op()
    pop = msa.popitem() if True else tf.no_op()
    size = msa.size()
    clear = msa.clear()

init_op = tf.global_variables_initializer()

config = tf.ConfigProto(allow_soft_placement=False)

with tf.Session(config=config) as S:
    S.run(init_op)
    assert S.run(size) == 0

    S.run(stage1)
    assert S.run(size) == 1

    S.run(stage2)
    assert S.run(size) == 2

    print S.run(get)
    assert S.run(size) == 2

    print S.run(unstage)
    assert S.run(size) == 1

    print S.run(pop)
    assert S.run(size) == 0

    S.run(stage1)
    assert S.run(size) == 1

    S.run(clear)
    assert S.run(size) == 0