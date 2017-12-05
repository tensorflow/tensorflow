# min quantize
a quantize&obfuscate library for tensorflow graph

## Usage
in case of a graph with following structure:

```python
# for Variable, will be converted to constant by quantize
w = tf.Variable(..., name='w')
b = tf.constant(..., name='b')
x = tf.placeholder(..., name='x')
y = tf.add(tf.matmul(w, x), b, name='y')
```

now we got a simple $y=Wx+b$ structure

usually we will got some training code, let's add them

```python
optm = tf.train.AdamOptimizer()
label = tf.placeholder(...)
cost = tf.sqrt(tf.reduce_mean(tf.square(label - y)))
train_op = optm.minimize(cost)
```

then, let's extract the graph

```python
g = tf.get_default_graph().as_graph_def()
with open('graph.pb', 'wb') as fp:
	fp.write(g.SerializeToString())
```

now, use out lib

```bash
bazel run //tensorflow/tools/min_quantize:quantize -- \
	--input graph.pb --output quantized.pb --output_node y
```

we will got a quantized graph ```quantized.pb```, only contains our desired $y=Wx+b$ structure, strip useless part, convert ```w``` to constant.

then, obfuscate it

```bash
bazel run //tensorflow/tools/min_quantize:obfuscate -- \
	--input quantized.pb --quantized \
	--output obfuscated.pb \
	--output_mapping mapping.txt \
	--keep x --keep y
```

we will got a obfuscated graph ```obfuscated.pb```, ```--quantized``` flag tell obfuscate tool that ```quantized.pb``` is a quantized graph, no a raw tensorflow graph. obfuscate mapping informating stored in mapping.txt

```
x:x
b:a
w:b
w/read:c
MatMul:d
y:y
```

or we could just obfuscated a raw tensorflow graph

```bash
bazel run //tensorflow/tools/min_quantize:obfuscate -- \
	--input graph.pb \
	--output obfuscated_raw_graph.pb \
	--output_mapping mapping.txt \
	--keep x --keep y
```

we could use them in both python or c++

```python
from tensorflow.contrib.min_quantize.quantize_lib import feeds_of_graph, load_graph
from copy import copy
# load graph
quantized_graph = load_graph('quantized.pb')
# restore quantized data
feeds = feeds_of_graph(quantized_graph)
with tf.Session() as s:
	# import
	tf.import_graph_def(quantized_graph.graph, name='')
	# shadow copy feeds
	feed_dict = copy(feeds)
	# put x
	feed_dict['x:0'] = ...
	# run
	y = s.run('y:0', feed_dict)
```

with c++ (android available)

change bazel build script:

* add deps ``` //tensorflow/contrib/min_quantize:protos_all_cc ```
* add source ``` //tensorflow/contrib/min_quantize:quantized_graph_loader ```

```c++
#include "tensorflow/contrib/min_quantize/quantized_graph_loader.h"

// assume we have init session as follow
tensorflow::Session session;
GraphData *graph_data;

// load, please check returned Status
loadQuantizedGraph(&session, &graph_data, [input stream or bytes]);

// now, we got a input as follow
std::vector<std::pair<std::string, tensorflow::Tensor> > inputs;

// put quantized data in inputs
graph_data->append(inputs);

// now use inputs as usual, eg:
inputs.push_back(std::make_pair("x", ...));
std::vector<tensorflow::Tensor> output;
session.Run(inputs, {"y"}, {}, &output);
```

## Features
* quantize library without Quantize/Dequantize Ops
* use KMeans for better quantize accuracy
* obfuscate graph node name, prevent easily graph extraction
