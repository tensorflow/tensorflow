# Dataset Input Pipeline
[TOC]

@{tf.data.Dataset} allows you to build complex input pipelines. See the
@{$datasets$programmer's guide} for an in-depth explanation of how to use this
API.

## Reader classes

Classes that create a dataset from input files.

*   @{tf.data.FixedLengthRecordDataset}
*   @{tf.data.TextLineDataset}
*   @{tf.data.TFRecordDataset}

## Creating new datasets

Static methods in `Dataset` that create new datasets.

*   @{tf.data.Dataset.from_generator}
*   @{tf.data.Dataset.from_sparse_tensor_slices}
*   @{tf.data.Dataset.from_tensor_slices}
*   @{tf.data.Dataset.from_tensors}
*   @{tf.data.Dataset.list_files}
*   @{tf.data.Dataset.range}
*   @{tf.data.Dataset.zip}

## Transformations on existing datasets

These functions transform an existing dataset, and return a new dataset. Calls
can be chained together, as shown in the example below:

```
train_data = train_data.batch(100).shuffle().repeat()
```

*   @{tf.data.Dataset.apply}
*   @{tf.data.Dataset.batch}
*   @{tf.data.Dataset.cache}
*   @{tf.data.Dataset.concatenate}
*   @{tf.data.Dataset.filter}
*   @{tf.data.Dataset.flat_map}
*   @{tf.data.Dataset.interleave}
*   @{tf.data.Dataset.map}
*   @{tf.data.Dataset.padded_batch}
*   @{tf.data.Dataset.prefetch}
*   @{tf.data.Dataset.repeat}
*   @{tf.data.Dataset.shard}
*   @{tf.data.Dataset.shuffle}
*   @{tf.data.Dataset.skip}
*   @{tf.data.Dataset.take}

### Custom transformation functions

Custom transformation functions can be applied to a `Dataset` using @{tf.data.Dataset.apply}. Below are custom transformation functions from `tf.contrib.data`:

*   @{tf.contrib.data.batch_and_drop_remainder}
*   @{tf.contrib.data.dense_to_sparse_batch}
*   @{tf.contrib.data.enumerate_dataset}
*   @{tf.contrib.data.group_by_window}
*   @{tf.contrib.data.ignore_errors}
*   @{tf.contrib.data.rejection_resample}
*   @{tf.contrib.data.sloppy_interleave}
*   @{tf.contrib.data.unbatch}

## Iterating over datasets

These functions make a @{tf.data.Iterator} from a `Dataset`.

*   @{tf.data.Dataset.make_initializable_iterator}
*   @{tf.data.Dataset.make_one_shot_iterator}

The `Iterator` class also contains static methods that create a @{tf.data.Iterator} that can be used with multiple `Dataset` objects.

*   @{tf.data.Iterator.from_structure}
*   @{tf.data.Iterator.from_string_handle}

## Extra functions from `tf.contrib.data`

*   @{tf.contrib.data.read_batch_features}

