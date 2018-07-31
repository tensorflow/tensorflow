# Bigtable #

[Cloud Bigtable](https://cloud.google.com/bigtable/) is a high
performance storage system that can store and serve training data. This contrib
package contains an experimental integration with TensorFlow.

> **Status: Highly experimental.** The current implementation is very much in
> flux. Please use at your own risk! :-)

The TensorFlow integration with Cloud Bigtable is optimized for common
TensorFlow usage and workloads. It is currently optimized for reading from Cloud
Bigtable at high speed, in particular to feed modern accelerators. For
general-purpose Cloud Bigtable
APIs, see the [official Cloud Bigtable client library documentation][clientdoc].

[clientdoc]:  https://cloud.google.com/bigtable/docs/reference/libraries

## Sample Use

There are three main reading styles supported by the `BigtableTable` class:

 1. **Reading keys**: Read only the row keys in a table. Keys are returned in
    sorted order from the table. Most key reading operations retrieve all keys
    in a contiguous range, however the `sample_keys` operation skips keys, and
    operates on the whole table (and not a contiguous subset).
 2. **Retrieving a row's values**: Given a row key, look up the data associated
    with a defined set of columns. This operation takes advantage of Cloud
    Bigtable's low-latency and excellent support for random access.
 3. **Scanning ranges**: Given a contiguous range of rows retrieve both the row
    key and the data associated with a fixed set of columns. This operation
    takes advantage of Cloud Bigtable's high throughput scans, and is the most
    efficient way to read data.

When using the Cloud Bigtable API, the workflow is:

 1. Create a `BigtableClient` object.
 2. Use the `BigtableClient` to create `BigtableTable` objects corresponding to
    each table in the Cloud Bigtable instance you would like to access.
 3. Call methods on the `BigtableTable` object to create `tf.data.Dataset`s to
    retrieve data.

The following is an example for how to read all row keys with the prefix
`train-`.

```python
import tensorflow as tf

GCP_PROJECT_ID = '<FILL_ME_IN>'
BIGTABLE_INSTANCE_ID = '<FILL_ME_IN>'
BIGTABLE_TABLE_NAME = '<FILL_ME_IN>'
PREFIX = 'train-'

def main():
  client = tf.contrib.cloud.BigtableClient(GCP_PROJECT_ID, BIGTABLE_INSTANCE_ID)
  table = client.table(BIGTABLE_TABLE_NAME)
  dataset = table.keys_by_prefix_dataset(PREFIX)
  iterator = dataset.make_initializable_iterator()
  get_next_op = iterator.get_next()

  with tf.Session() as sess:
    print('Initializing the iterator.')
    sess.run(iterator.initializer)
    print('Retrieving rows:')
    row_index = 0
    while True:
      try:
        row_key = sess.run(get_next_op)
        print('Row key %d: %s' % (row_index, row_key))
        row_index += 1
      except tf.errors.OutOfRangeError:
        print('Finished reading data!')
        break

if __name__ == '__main__':
  main()

```

### Reading row keys

Read only the row keys in a table. Keys are returned in sorted order from the
table. Most key reading operations retrieve all keys in a contiguous range,
however the `sample_keys` operation skips keys, and operates on the whole table
(and not a contiguous subset).

There are 3 methods to retrieve row keys:

 - `table.keys_by_range_dataset(start, end)`: Retrieve row keys starting with
   `start`, and ending with `end`. The range is "half-open", and thus it
   includes `start` if `start` is present in the table. It does not include
   `end`.
 - `table.keys_by_prefix_dataset(prefix)`: Retrieves all row keys that start
   with `prefix`. It includes the row key `prefix` if present in the table.
 - `table.sample_keys()`: Retrieves a sampling of keys from the underlying
   table. This is often useful in conjunction with parallel scans.

### Reading cell values given a row key

Given a dataset producing row keys, you can use the `table.lookup_columns`
transformation to retrieve values. Example:

```python
key_dataset = tf.data.Dataset.from_tensor_slices([
    'row_key_1',
    'other_row_key',
    'final_row_key',
])
values_dataset = key_dataset.apply(
  table.lookup_columns(('my_column_family', 'column_name'),
                       ('other_cf', 'col')))
training_data = values_dataset.map(my_parsing_function)  # ...
```

### Scanning ranges
Given a contiguous range of rows retrieve both the row key and the data
associated with a fixed set of columns. Scanning is the most efficient way to
retrieve data from Cloud Bigtable and is thus a very common API for high
performance data pipelines. To construct a scanning `tf.data.Dataset` from a
`BigtableTable` object, call one of the following methods:

 - `table.scan_prefix(prefix, ...)`
 - `table.scan_range(start, end, ...)`
 - `table.parallel_scan_prefix(prefix, ...)`
 - `table.parallel_scan_range(start, end, ...)`

Aside from the specification of the contiguous range of rows, they all take the
following arguments:

 - `probability`: (Optional.) A float between 0 (exclusive) and 1 (inclusive).
      A non-1 value indicates to probabilistically sample rows with the
      provided probability.
 - `columns`: The columns to read. (See below.)
 - `**kwargs`: The columns to read. (See below.)

In addition the two parallel operations accept the following optional argument:
`num_parallel_scans` which configures the number of parallel Cloud Bigtable scan
operations to run. A reasonable default is automatically chosen for small
Cloud Bigtable clusters. If you have a large cluster, or an extremely demanding
workload, you can tune this value to optimize performance.

#### Specifying columns to read when scanning

All of the scan operations allow you to specify the column family and columns
in the same ways.

##### Using `columns`

The first way to specify the data to read is via the `columns` parameter. The
value should be a tuple (or list of tuples) of strings. The first string in the
tuple is the column family, and the second string in the tuple is the column
qualifier.

##### Using `**kwargs`

The second way to specify the data to read is via the `**kwargs` parameter,
which you can use to specify keyword arguments corresponding to the columns that
you want to read. The keyword to use is the column family name, and the argument
value should be either a string, or a tuple of strings, specifying the column
qualifiers (column names).

Although using `**kwargs` has the advantage of requiring less typing, it is not
future-proof in all cases. (If we add a new parameter to the scan functions that
has the same name as your column family, your code will break.)

##### Examples

Below are two equivalent snippets for how to specify which columns to read:

```python
ds1 = table.scan_range("row_start", "row_end", columns=[("cfa", "c1"),
                                                        ("cfa", "c2"),
                                                        ("cfb", "c3")])
ds2 = table.scan_range("row_start", "row_end", cfa=["c1", "c2"], cfb="c3")
```

In this example, we are reading 3 columns from a total of 2 column families.
From the `cfa` column family, we are reading columns `c1`, and `c2`. From the
second column family (`cfb`), we are reading `c3`. Both `ds1` and `ds2` will
output elements of the following types (`tf.string`, `tf.string`, `tf.string`,
`tf.string`). The first `tf.string` is the row key, the second `tf.string` is
the latest data in cell `cfa:c1`, the third corresponds to `cfa:c2`, and the
final one is `cfb:c3`.

#### Determinism when scanning

While the non-parallel scan operations are fully deterministic, the parallel
scan operations are not. If you would like to scan in parallel without losing
determinism, you can build up the `parallel_interleave` yourself. As an example,
say we wanted to scan all rows between `training_data_00000`, and
`training_data_90000`, we can use the following code snippet:

```python
table = # ...
columns = [('cf1', 'col1'), ('cf1', 'col2')]
NUM_PARALLEL_READS = # ...
ds = tf.data.Dataset.range(9).shuffle(10)
def interleave_fn(index):
  # Given a starting index, create 2 strings to be the start and end
  start_idx = index
  end_idx = index + 1
  start_idx_str = tf.as_string(start_idx * 10000, width=5, fill='0')
  end_idx_str = tf.as_string(end_idx * 10000, width=5, fill='0')
  start = tf.string_join(['training_data_', start_idx_str])
  end = tf.string_join(['training_data_', end_idx_str])
  return table.scan_range(start_idx, end_idx, columns=columns)
ds = ds.apply(tf.contrib.data.parallel_interleave(
    interleave_fn, cycle_length=NUM_PARALLEL_READS, prefetch_input_elements=1))
```

> Note: you should divide up the key range into more sub-ranges for increased
> parallelism.

## Writing to Cloud Bigtable

In order to simplify getting started, this package provides basic support for
writing data into Cloud Bigtable.

> Note: The implementation is not optimized for performance! Please consider
> using alternative frameworks such as Apache Beam / Cloud Dataflow for
> production workloads.

Below is an example for how to write a trivial dataset into Cloud Bigtable.

```python
import tensorflow as tf

GCP_PROJECT_ID = '<FILL_ME_IN>'
BIGTABLE_INSTANCE_ID = '<FILL_ME_IN>'
BIGTABLE_TABLE_NAME = '<FILL_ME_IN>'
COLUMN_FAMILY = '<FILL_ME_IN>'
COLUMN_QUALIFIER = '<FILL_ME_IN>'

def make_dataset():
  """Makes a dataset to write to Cloud Bigtable."""
  return tf.data.Dataset.from_tensor_slices([
      'training_data_1',
      'training_data_2',
      'training_data_3',
  ])

def make_row_key_dataset():
  """Makes a dataset of strings used for row keys.

  The strings are of the form: `fake-data-` followed by a sequential counter.
  For example, this dataset would contain the following elements:

   - fake-data-00000001
   - fake-data-00000002
   - ...
   - fake-data-23498103
  """
  counter_dataset = tf.contrib.data.Counter()
  width = 8
  row_key_prefix = 'fake-data-'
  ds = counter_dataset.map(lambda index: tf.as_string(index,
                                                      width=width,
                                                      fill='0'))
  ds = ds.map(lambda idx_str: tf.string_join([row_key_prefix, idx_str]))
  return ds


def main():
  client = tf.contrib.cloud.BigtableClient(GCP_PROJECT_ID, BIGTABLE_INSTANCE_ID)
  table = client.table(BIGTABLE_TABLE_NAME)
  dataset = make_dataset()
  index_dataset = make_row_key_dataset()
  aggregate_dataset = tf.data.Dataset.zip((index_dataset, dataset))
  write_op = table.write(aggregate_dataset, column_families=[COLUMN_FAMILY],
                         columns=[COLUMN_QUALIFIER])

  with tf.Session() as sess:
    print('Starting transfer.')
    sess.run(write_op)
    print('Transfer complete.')

if __name__ == '__main__':
  main()
```

## Sample applications and architectures

While most machine learning applications are well suited by a high performance
distributed file system, there are certain applications where using Cloud
Bigtable works extremely well.

### Perfect Shuffling

Normally, training data is stored in flat files, and a combination of
(1) `tf.data.Dataset.interleave` (or `parallel_interleave`), (2)
`tf.data.Dataset.shuffle`, and (3) writing the data in an unsorted order in the
data files in the first place, provides enough randomization to ensure models
train efficiently. However, if you would like perfect shuffling, you can use
Cloud Bigtable's low-latency random access capabilities. Create a
`tf.data.Dataset` that generates the keys in a perfectly random order (or read
all the keys into memory and use a shuffle buffer sized to fit all of them for a
perfect random shuffle using `tf.data.Dataset.shuffle`), and then use
`lookup_columns` to retrieve the training data.

### Distributed Reinforcement Learning

Sophisticated reinforcement learning algorithms are commonly trained across a
distributed cluster. (See [IMPALA by DeepMind][impala].) One part of the cluster
runs self-play, while the other part of the cluster learns a new version of the
model based on the training data generated by self-play. The new model version
is then distributed to the self-play half of the cluster, and new training data
is generated to continue the cycle.

In such a configuration, because there is value in training on the freshest
examples, a storage service like Cloud Bigtable can be used to store and
serve the generated training data. When using Cloud Bigtable, there is no need
to aggregate the examples into large batch files, but the examples can instead
be written as soon as they are generated, and then retrieved at high speed.

[impala]: https://arxiv.org/abs/1802.01561

## Common Gotchas!

### gRPC Certificates

If you encounter a log line that includes the following:

```
"description":"Failed to load file", [...],
"filename":"/usr/share/grpc/roots.pem"
```

you likely need to copy the [gRPC roots.pem file][grpcPem] to
`/usr/share/grpc/roots.pem` on your local machine.

[grpcPem]: https://github.com/grpc/grpc/blob/master/etc/roots.pem

### Permission denied errors

The TensorFlow Cloud Bigtable client will search for credentials to use in the
process's environment. It will use the first credentials it finds if multiple
are available.

 - **Compute Engine**: When running on Compute Engine, the client will often use
   the service account from the virtual machine's metadata service. Be sure to
   authorize your Compute Engine VM to have access to the Cloud Bigtable service
   when creating your VM.
 - **Cloud TPU**: Your Cloud TPUs run with the designated Cloud TPU service
   account dedicated to your GCP project. Ensure the service account has been
   authorized via the Cloud Console to access your Cloud Bigtable instances.
