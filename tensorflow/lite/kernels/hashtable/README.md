# How to use TF Lookup ops in TFLite

The objective of this file is to provide examples to demonstrate how to use TF
Lookup ops in TFLite.

## Supported Tensorflow Lookup ops in TFLite

Here is the supported status of TensorFlow Lookup ops.

<table>
  <tr>
   <td><strong><em>TF Python lookup ops</em></strong>
   </td>
   <td colspan="5" ><strong><em>Supported status</em></strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" >tf.lookup.StaticHashTable
   </td>
   <td rowspan="2" colspan="5" >Supported only with tensor initializers.
<p>
Supported mapping type: string → int64, int64 → string
   </td>
  </tr>
  <tr>
  </tr>
  <tr>
   <td rowspan="2" >tf.lookup.Hashtable
   </td>
   <td rowspan="2" colspan="5" >Supported only with tensor initializers.
<p>
Supported mapping type: string → int64, int64 → string
   </td>
  </tr>
  <tr>
  </tr>
  <tr>
   <td rowspan="2" >tf.lookup.index_to_string_table_from_tensor
   </td>
   <td rowspan="2" colspan="5" >Supported.
   </td>
  </tr>
  <tr>
  </tr>
  <tr>
   <td rowspan="2" >tf.lookup.index_table_from_tensor
   </td>
   <td rowspan="2" colspan="5" >Supported natively when num_oov_bukcets=0 and dtype=dtypes.string.
<p>
For the oov concept, you will need a <a href="https://www.tensorflow.org/lite/guide/ops_select" title="Select TensorFlow operators to use in TensorFlow Lite">Flex delegate</a>.
   </td>
  </tr>
  <tr>
  </tr>
  <tr>
   <td>tf.lookup.StaticVocabularyTable
   </td>
   <td colspan="5" >Supported but you will need a <a href="https://www.tensorflow.org/lite/guide/ops_select" title="Select TensorFlow operators to use in TensorFlow Lite">Flex delegate</a>.
<p>
Use tf.index_table_from_tensor or tf.index_to_string_table_from_tensor instead if possible if you don’t want to use <a href="https://www.tensorflow.org/lite/guide/ops_select" title="Select TensorFlow operators to use in TensorFlow Lite">Flex delegate</a>.
   </td>
  </tr>
  <tr>
   <td>tf.lookup.experimental.DenseHashTable
<p>
tf.contrib.lookup.MutableHashTable
<p>
tf.contrib.lookup.MutableDenseHashTable
   </td>
   <td colspan="5" >Not supported yet.
   </td>
  </tr>
  <tr>
   <td>tf.lookup.IdTableWithHashBuckets
   </td>
   <td colspan="5" >Supported but you need a <a href="https://www.tensorflow.org/lite/guide/ops_select" title="Select TensorFlow operators to use in TensorFlow Lite">Flex delegate</a>.
   </td>
  </tr>
</table>



## Python Sample code

Here, you can find the Python sample code:



*   Static hash table (string → int64)

```
int64_values = tf.constant([1, 2, 3], dtype=tf.int64)
string_values = tf.constant(['bar', 'foo', 'baz'], dtype=tf.string)

initializer = tf.lookup.KeyValueTensorInitializer(string_values, int64_values)
table = tf.lookup.StaticHashTable(initializer, 4)

with tf.control_dependencies([tf.initializers.tables_initializer()]):
  input_string_tensor = tf.compat.v1.placeholder(tf.string, shape=[1])
  out_int64_tensor = table.lookup(input_string_tensor)
```

*   Static hash table, initialized from a file (string → int64)

```
with open('/tmp/vocab.file', 'r') as f:
  words = f.read().splitlines()

string_values = tf.constant(words, dtype=tf.string)

initializer = tf.lookup.KeyValueTensorInitializer(string_values, int64_values)
table = tf.lookup.StaticHashTable(initializer, 4)

with tf.control_dependencies([tf.initializers.tables_initializer()]):
  input_string_tensor = tf.placeholder(tf.string, shape=[1])
  out_int64_tensor = table.lookup(input_string_tensor)
```

*   Index table (string → int64)

```
UNK_ID = -1
vocab = tf.constant(["emerson", "lake", "palmer"])
vocab_table = tf.lookup.index_table_from_tensor(vocab, default_value=UNK_ID)

input_tensor = tf.compat.v1.placeholder(tf.string, shape=[5])

with tf.control_dependencies([tf.initializers.tables_initializer()]):
  out_tensor = vocab_table.lookup(input_tensor)
```

*   Index table, initialized from a file (string → int64)

```
with open('/tmp/vocab.file', 'r') as f:
  words = f.read().splitlines()

UNK_ID = -1
vocab = tf.constant(words)
vocab_table = tf.lookup.index_table_from_tensor(vocab, default_value=UNK_ID)

input_tensor = tf.compat.v1.placeholder(tf.string, shape=[5])

with tf.control_dependencies([tf.initializers.tables_initializer()]):
  out_tensor = vocab_table.lookup(input_tensor)
```

*   Index to string table (int64 → string)

```
UNK_WORD = "unknown"
vocab = tf.constant(["emerson", "lake", "palmer"])
vocab_table = tf.lookup.index_to_string_table_from_tensor(vocab, default_value=UNK_WORD)

input_tensor = tf.compat.v1.placeholder(tf.int64, shape=[1])

with tf.control_dependencies([tf.initializers.tables_initializer()]):
  out_tensor = vocab_table.lookup(input_tensor)
```

*   Index to string table, initialized from a file (int64 → string)

```
with open('/tmp/vocab.file', 'r') as f:
  words = f.read().splitlines()

UNK_WORD = "unknown"
vocab = tf.constant(words)
vocab_table = tf.lookup.index_to_string_table_from_tensor(vocab, default_value=UNK_WORD)

input_tensor = tf.compat.v1.placeholder(tf.int64, shape=[1])

with tf.control_dependencies([tf.initializers.tables_initializer()]):
  out_tensor = vocab_table.lookup(input_tensor)
```

## How to Include Hashtable ops in your TFLite.

Currently, hashtable ops are not included in the builtin op set. You need to add
hashtable ops manually by including the following dependency:

`"//tensorflow/lite/kernels/hashtable:hashtable_op_kernels"`

And then, your op resolver should add them like the following statements:


```
  // Add hashtable op handlers.
  tflite::ops::custom::AddHashtableOps(&resolver);
```
