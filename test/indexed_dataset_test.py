import tensorflow as tf
import numpy as np
import sys
import pytest
import glob
import os
import time

def get_ckpt_dir():
    return "/tmp/eparallax-{}/checkpoint/index/".format(os.environ["USER"])

def do_initialize_ckpt():
    for f in glob.glob(get_ckpt_dir() + "*"):
        os.remove(f)

def initialize_ckpt(fn):
    def wrapper(*args, **kwargs):
        do_initialize_ckpt()
        return fn(*args, **kwargs)
    return wrapper

def aggregate_ckpt(fn):
    def wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)
        with open(get_ckpt_dir() + "index_ckpt", 'a') as global_ckpt:
            for f in glob.glob(get_ckpt_dir() + "index_ckpt_*"):
                with open(f, 'r') as shard_ckpt:
                    global_ckpt.write(shard_ckpt.read())
                os.remove(f)
        return ret
    return wrapper

@aggregate_ckpt
def run_steps(graph, n, num_steps, initializer=None):
    with tf.compat.v1.Session(graph=graph) as sess:
        if initializer is not None:
            sess.run(initializer)
        res = []
        for _ in range(num_steps):
            res.append(sess.run(n))
        return res

@initialize_ckpt
def test_tensor_slices():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,1,2,3,4]
    res = run_steps(g, n, 4)
    assert res == [5,6,7,8]
    res = run_steps(g, n, 1)
    assert res == [9]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_flat_map():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices([[0,1,2],[3,4,5],[6,7,8],
                                                 [9,10,11],[12,13,14],[15,16,17],
                                                 [18,19,20],[21,22,23],[24,25,26]])
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,1,2,3,4]
    res = run_steps(g, n, 4)
    assert res == [5,6,7,8]
    res = run_steps(g, n, 1)
    assert res == [9]
    res = run_steps(g, n, 9)
    assert res == [10, 11, 12, 13, 14, 15,16,17, 18]
    res = run_steps(g, n, 4)
    assert res == [19,20,21,22]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 5)

@initialize_ckpt
def test_multi_level_flat_map():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices([[[0,1,2],[3,4,5],[6,7,8]],
                                                 [[9,10,11],[12,13,14],[15,16,17]],
                                                 [[18,19,20],[21,22,23],[24,25,26]]])
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,1,2,3,4]
    res = run_steps(g, n, 4)
    assert res == [5,6,7,8]
    res = run_steps(g, n, 1)
    assert res == [9]
    res = run_steps(g, n, 9)
    assert res == [10,11,12,13,14,15,16,17,18]
    res = run_steps(g, n, 4)
    assert res == [19,20,21,22]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 5)

@initialize_ckpt
def test_interleave():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices([[0,1,2],[3,4,5],[6,7,8],
                                                 [9,10,11],[12,13,14],[15,16,17],
                                                 [18,19,20],[21,22,23],[24,25,26]])
        ds = ds.interleave(tf.data.Dataset.from_tensor_slices)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,3,6,9,12]
    res = run_steps(g, n, 4)
    assert res == [15,18,21,24]
    res = run_steps(g, n, 6)
    assert res == [1,4,7,10,13,16]
    res = run_steps(g, n, 9)
    assert res == [19,22,25,2,5,8,11,14,17]
    res = run_steps(g, n, 3)
    assert res == [20,23,26]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_multi_level_interleave():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices([[[0,1,2],[3,4,5],[6,7,8]],
                                                 [[9,10,11],[12,13,14],[15,16,17]],
                                                 [[18,19,20],[21,22,23],[24,25,26]]])
        ds = ds.interleave(tf.data.Dataset.from_tensor_slices)
        ds = ds.interleave(tf.data.Dataset.from_tensor_slices)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,9,18,3,12]
    res = run_steps(g, n, 4)
    assert res == [21,6,15,24]
    res = run_steps(g, n, 1)
    assert res == [1]
    res = run_steps(g, n, 9)
    assert res == [10,19,4,13,22,7,16,25,2]
    res = run_steps(g, n, 4)
    assert res == [11,20,5,14]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 5)

@initialize_ckpt
def test_map():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
        ds = ds.map(lambda x: 2*x)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,2,4,6,8]
    res = run_steps(g, n, 4)
    assert res == [10,12,14,16]
    res = run_steps(g, n, 1)
    assert res == [18]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_textline():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.TextLineDataset('example_data_0.txt')
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert [int(r) for r in res] == [0,1,2,3,4]
    res = run_steps(g, n, 5)
    assert [int(r) for r in res] == [5,6,7,8,9]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_experimental_parallel_interleave():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.list_files(
                ['example_data_0.txt', 'example_data_1.txt'], shuffle=False)
        ds = ds.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TextLineDataset, cycle_length=1))
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert [int(r) for r in res] == [0,1,2,3,4]
    res = run_steps(g, n, 4)
    assert [int(r) for r in res] == [5,6,7,8]
    res = run_steps(g, n, 1)
    assert [int(r) for r in res] == [9]
    res = run_steps(g, n, 9)
    assert [int(r) for r in res] == [10,11,12,13,14,15,16,17,18]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 4)

@initialize_ckpt
def test_batch():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(50)))
        ds = ds.batch(5)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 1)
    expected_result = [np.array(list(range(i*5, i*5+5))) for i in range(0, 1)]
    assert len(res) == len(expected_result)
    assert all([(r == e).all() for r, e in zip(res, expected_result)])
    res = run_steps(g, n, 5)
    expected_result = [np.array(list(range(i*5, i*5+5))) for i in range(1, 6)]
    assert len(res) == len(expected_result)
    assert all([(r == e).all() for r, e in zip(res, expected_result)])
    res = run_steps(g, n, 3)
    expected_result = [np.array(list(range(i*5, i*5+5))) for i in range(6, 9)]
    assert len(res) == len(expected_result)
    assert all([(r == e).all() for r, e in zip(res, expected_result)])
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 2)

@initialize_ckpt
def test_multi_level_batch():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(100)))
        ds = ds.batch(5)
        ds = ds.batch(2)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 1)
    expected_result = [np.array([list(range(i*5, i*5+5))
            for i in range(j*2, j*2+2)]) for j in range(0, 1)]
    assert len(res) == len(expected_result)
    assert all([(r == e).all() for r, e in zip(res, expected_result)])
    res = run_steps(g, n, 5)
    expected_result = [np.array([list(range(i*5, i*5+5))
            for i in range(j*2, j*2+2)]) for j in range(1, 6)]
    assert len(res) == len(expected_result)
    assert all([(r == e).all() for r, e in zip(res, expected_result)])
    res = run_steps(g, n, 3)
    expected_result = [np.array([list(range(i*5, i*5+5))
            for i in range(j*2, j*2+2)]) for j in range(6, 9)]
    assert len(res) == len(expected_result)
    assert all([(r == e).all() for r, e in zip(res, expected_result)])
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 2)

@initialize_ckpt
def test_zip():
    g = tf.Graph()
    with g.as_default():
        ds1 = tf.data.Dataset.from_tensor_slices(list(range(10)))
        ds2 = tf.data.Dataset.from_tensor_slices(list(range(9,-1,-1)))
        ds = tf.data.Dataset.zip((ds1, ds2))
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 2)
    assert res == [(0, 9), (1, 8)]
    res = run_steps(g, n, 3)
    assert res == [(2, 7), (3, 6), (4, 5)]
    res = run_steps(g, n, 4)
    assert res == [(5, 4), (6, 3), (7, 2), (8, 1)]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 2)

@initialize_ckpt
def test_shuffle():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(50)))
        ds = ds.shuffle(10)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = []
    res += run_steps(g, n, 5)
    res += run_steps(g, n, 5)
    res += run_steps(g, n, 6)
    res += run_steps(g, n, 7)
    res += run_steps(g, n, 8)
    res += run_steps(g, n, 9)
    res += run_steps(g, n, 10)
    assert sorted(res) == list(range(50))
    assert list(set(sorted(res))) == sorted(res) # Assert no duplicate element
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_prefetch():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(20)))
        ds = ds.prefetch(20)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,1,2,3,4]
    res = run_steps(g, n, 5)
    assert res == [5,6,7,8,9]
    res = run_steps(g, n, 5)
    assert res == [10,11,12,13,14]
    res = run_steps(g, n, 5)
    assert res == [15,16,17,18,19]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_forever_repeat():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
        ds = ds.repeat()
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    for _ in range(5):
        res = run_steps(g, n, 5)
        assert res == [0,1,2,3,4]
        res = run_steps(g, n, 5)
        assert res == [5,6,7,8,9]
        res = run_steps(g, n, 3)
        assert res == [0,1,2]
        res = run_steps(g, n, 6)
        assert res == [3,4,5,6,7,8]
        res = run_steps(g, n, 4)
        assert res == [9,0,1,2]
        res = run_steps(g, n, 7)
        assert res == [3,4,5,6,7,8,9]

@initialize_ckpt
def test_finite_repeat():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
        ds = ds.repeat(3)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,1,2,3,4]
    res = run_steps(g, n, 5)
    assert res == [5,6,7,8,9]
    res = run_steps(g, n, 3)
    assert res == [0,1,2]
    res = run_steps(g, n, 6)
    assert res == [3,4,5,6,7,8]
    res = run_steps(g, n, 4)
    assert res == [9,0,1,2]
    res = run_steps(g, n, 7)
    assert res == [3,4,5,6,7,8,9]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_repeat_and_shuffle():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
        ds = ds.repeat(3)
        ds = ds.shuffle(15)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = []
    res = run_steps(g, n, 5)
    res += run_steps(g, n, 5)
    res += run_steps(g, n, 3)
    res += run_steps(g, n, 6)
    res += run_steps(g, n, 4)
    res += run_steps(g, n, 7)
    assert sorted(res) == sorted(list(range(10)) * 3)
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_shuffle_and_repeat():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
        ds = ds.shuffle(15)
        ds = ds.repeat(3)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = []
    res = run_steps(g, n, 5)
    res += run_steps(g, n, 5)
    res += run_steps(g, n, 3)
    res += run_steps(g, n, 6)
    res += run_steps(g, n, 4)
    res += run_steps(g, n, 7)
    assert sorted(res) == sorted(list(range(10)) * 3)
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_shard():
    g = tf.Graph()
    with g.as_default():
        ds1 = tf.data.Dataset.from_tensor_slices(list(range(20)))
        ds2 = tf.data.Dataset.from_tensor_slices(list(range(20)))
        ds1 = ds1.shard(2, 0)
        ds2 = ds2.shard(2, 1)
        n1 = tf.compat.v1.data.make_one_shot_iterator(ds1).get_next()
        n2 = tf.compat.v1.data.make_one_shot_iterator(ds2).get_next()
    res = run_steps(g, [n1, n2], 5)
    assert res == [[0,1],[2,3],[4,5],[6,7],[8,9]]
    res = run_steps(g, [n1, n2], 4)
    assert res == [[10,11],[12,13],[14,15],[16,17]]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, [n1, n2], 2)

@initialize_ckpt
def test_cache():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
        ds = ds.cache()
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,1,2,3,4]
    res = run_steps(g, n, 4)
    assert res == [5,6,7,8]
    res = run_steps(g, n, 1)
    assert res == [9]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_tf_record():
    pass

@initialize_ckpt
def test_take():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
        ds = ds.take(5)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 1)
    assert res == [0]
    res = run_steps(g, n, 3)
    assert res == [1,2,3]
    res = run_steps(g, n, 1)
    assert res == [4]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_tf_record_iterleave():
    g = tf.Graph()
    with g.as_default():
        file_names = [
            "/cmsdata/ssd1/cmslab/imagenet-data/aws/train-00{}-of-01024"
            .format(i) for i in range(256, 256+128)
        ]
        ds = tf.data.TFRecordDataset.list_files(file_names)
        ds = ds.interleave(tf.data.TFRecordDataset)
        iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
        n = iterator.get_next()

    run_steps(g, n, 1)

@initialize_ckpt
def test_tf_record_experimental_iterleave():
    g = tf.Graph()
    with g.as_default():
        file_names = [
            "/cmsdata/ssd1/cmslab/imagenet-data/aws/train-00{}-of-01024".format(i) for i in range(256, 256+128)
        ]
        ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)
        ds = ds.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=1))
        iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
        n = iterator.get_next()

    run_steps(g, n, 10)

@initialize_ckpt
def test_interleave_and_prefetch():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices([[0,1,2],[3,4,5],[6,7,8],
                                                 [9,10,11],[12,13,14],[15,16,17],
                                                 [18,19,20],[21,22,23],[24,25,26]])
        ds = ds.interleave(tf.data.Dataset.from_tensor_slices)
        ds = ds.prefetch(10)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,3,6,9,12]
    res = run_steps(g, n, 4)
    assert res == [15,18,21,24]
    res = run_steps(g, n, 6)
    assert res == [1,4,7,10,13,16]
    res = run_steps(g, n, 9)
    assert res == [19,22,25,2,5,8,11,14,17]
    res = run_steps(g, n, 3)
    assert res == [20,23,26]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_imagenet():
    g = tf.Graph()
    with g.as_default():
        file_names = [
            "/cmsdata/ssd1/cmslab/imagenet-data/aws/train-00{}-of-01024".format(i) for i in range(256, 256+4)
        ]
        file_names.sort()
        batch_size = 16
        num_splits = 1
        batch_size_per_split = batch_size // num_splits
        num_workers = 2
        worker_id = 0
        ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)
        ds = ds.shard(num_workers, worker_id)
        ds = ds.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=10))
        counter = tf.data.Dataset.range(batch_size)
        counter = counter.repeat()
        ds = tf.data.Dataset.zip((ds, counter))
        ds = ds.prefetch(buffer_size=batch_size)
        ds = ds.shuffle(buffer_size=50, seed=2020)
        ds = ds.repeat()
        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                map_func=lambda *x:x,
                batch_size=batch_size_per_split,
                num_parallel_batches=num_splits))
        ds = ds.prefetch(buffer_size=num_splits)
        iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
        n = iterator.get_next()

    expected_result = run_steps(g, n, 100)
    do_initialize_ckpt()
    res = run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    res += run_steps(g, n, 10)
    assert all([all(o[0] == r[0]) and all(o[1] == r[1])
            for o, r in zip(expected_result, res)])

@initialize_ckpt
def test_prefetch_and_prefetch():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices(list(range(20)))
        ds = ds.prefetch(10)
        ds = ds.prefetch(10)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,1,2,3,4]
    res = run_steps(g, n, 5)
    assert res == [5,6,7,8,9]
    res = run_steps(g, n, 5)
    assert res == [10,11,12,13,14]
    res = run_steps(g, n, 5)
    assert res == [15,16,17,18,19]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_parallel_interleave():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices([[0,1,2],[3,4,5],[6,7,8],
                                                 [9,10,11],[12,13,14],[15,16,17],
                                                 [18,19,20],[21,22,23],[24,25,26]])
        ds = ds.interleave(tf.data.Dataset.from_tensor_slices,
                           num_parallel_calls=4)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 5)
    assert res == [0,3,6,9,12]
    res = run_steps(g, n, 4)
    assert res == [15,18,21,24]
    res = run_steps(g, n, 6)
    assert res == [1,4,7,10,13,16]
    res = run_steps(g, n, 9)
    assert res == [19,22,25,2,5,8,11,14,17]
    res = run_steps(g, n, 3)
    assert res == [20,23,26]
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_experimental_map_and_batch():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices([0,1,2,3,4,5,6,7,8,
                                                 9,10,11,12,13,14,15,16,17,
                                                 18,19,20,21,22,23,24,25,26])
        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                map_func=lambda x:10*x,
                batch_size=5,
                num_parallel_batches=2))
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    res = run_steps(g, n, 1)
    expected_result = [np.array([0,10,20,30,40])]
    assert all(all(r == e) for r, e in zip(res, expected_result))
    res = run_steps(g, n, 2)
    expected_result = [np.array([50,60,70,80,90]),np.array([100,110,120,130,140])]
    assert all(all(r == e) for r, e in zip(res, expected_result))
    res = run_steps(g, n, 2)
    expected_result = [np.array([150,160,170,180,190]),np.array([200,210,220,230,240])]
    assert all(all(r == e) for r, e in zip(res, expected_result))
    res = run_steps(g, n, 1)
    expected_result = [np.array([250,260])]
    assert all(all(r == e) for r, e in zip(res, expected_result))
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)

@initialize_ckpt
def test_map_and_batch():
    g = tf.Graph()
    with g.as_default():
        ds = tf.data.Dataset.from_tensor_slices([0,1,2,3,4,5,6,7,8,
                                                 9,10,11,12,13,14,15,16,17,
                                                 18,19,20,21,22,23,24,25,26])
        ds = ds.map(lambda x:10*x)
        ds = ds.batch(5)
        n = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()

    res = run_steps(g, n, 1)
    expected_result = [np.array([0,10,20,30,40])]
    assert all(all(r == e) for r, e in zip(res, expected_result))
    res = run_steps(g, n, 2)
    expected_result = [np.array([50,60,70,80,90]),np.array([100,110,120,130,140])]
    assert all(all(r == e) for r, e in zip(res, expected_result))
    res = run_steps(g, n, 2)
    expected_result = [np.array([150,160,170,180,190]),np.array([200,210,220,230,240])]
    assert all(all(r == e) for r, e in zip(res, expected_result))
    res = run_steps(g, n, 1)
    expected_result = [np.array([250,260])]
    assert all(all(r == e) for r, e in zip(res, expected_result))
    with pytest.raises(tf.errors.OutOfRangeError):
        res = run_steps(g, n, 1)
