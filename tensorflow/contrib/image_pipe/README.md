## ImagePipe for Tensorflow: an extremely-fast data input ops to GPU device (directly generation with ZeroCopy)

### With full RAW images stored in SSD to train ImageNet Resnet50 models, it could achieve ~96% performance of synthetic dataset training, which is faster than ImageDataGenerator and more simple than tf.TFRecord.

### Better to work with Horovod for best Distributed Training performance.


1) Deterministic image input by configuration of `seed`, which is not supported by tf.keras.preprocessing.ImageDataGenerator;

2) Support direct image generation with either NCHW or NHWC format;

3) Support target image resize in place and interleaving generation;

4) Reference of internal image directory format -

```sh
/train/
    /class-monkey/
        aug_1.jpg
        aug_2.jpg
    /class-bird/
        aug_1.jpg
        aug_2.jpg
```

### The usage of ImagePipe is similar to tf.keras.preprocessing.ImageDataGenerator
### Example of using ImagePipe:

```sh

import tensorflow as tf
from tensorflow.contrib.image_pipe.ops import gen_image_pipe_ops as image_pipe
import os

print('Download Raw JPEG images..')
os.system('curl -L https://github.com/ghostplant/lite-dnn/releases/download/lite-dataset/images-mnist.tar.gz | tar xzvf - -C /tmp >/dev/null')

print('Pipeline Raw JPEG images from disk to GPU with ZeroCopy..')
images, labels = image_pipe.image_pipe(directory_url='/tmp/train/',
    image_format='NCHW', batch_size=32, height=28, width=28,
    logging=True, seed=0, rescale=1.0/255, parallel=8, cache_size=1024)

out = tf.reshape(images, (-1, 3, 28, 28))
out = tf.layers.flatten(out)
out = tf.layers.dense(out, 512, activation=tf.nn.relu)
out = tf.layers.dropout(out)
out = tf.layers.dense(out, 512, activation=tf.nn.relu)
out = tf.layers.dropout(out)
out = tf.layers.dense(out, 10)

loss = tf.losses.sparse_softmax_cross_entropy(logits=out, labels=labels)
opt = tf.train.RMSPropOptimizer(0.0001, decay=1e-6).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int64), tf.argmax(out, 1)), tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10000):
    sess.run(opt)
    if i % 500 == 0:
      print('accuracy = %.2f %%' % (sess.run(accuracy) * 1e2))
  print('Done.')

```
