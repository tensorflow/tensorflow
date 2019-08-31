# Pre-computed receptive field parameters

## Table with results

The table below presents the receptive field parameters and cost (in terms of
floating point operations &mdash; FLOPs) for several popular convolutional
neural networks and their end-points. These are computed using the models from
the
[TF-Slim repository](https://github.com/tensorflow/models/tree/master/research/slim),
by using the
[rf_benchmark script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/receptive_field/python/util/examples/rf_benchmark.py).

Questions? See the [FAQ](#faq).

CNN                            | resolution | end-point            | FLOPs (Billion) | RF   | effective stride | effective padding
:----------------------------: | :--------: | :------------------: | :-------------: | :--: | :--------------: | :---------------:
alexnet_v2                     | None       | alexnet_v2/conv1     | None            | 11   | 4                | 0
alexnet_v2                     | None       | alexnet_v2/pool1     | None            | 19   | 8                | 0
alexnet_v2                     | None       | alexnet_v2/conv2     | None            | 51   | 8                | 16
alexnet_v2                     | None       | alexnet_v2/conv3     | None            | 99   | 16               | 32
alexnet_v2                     | None       | alexnet_v2/conv4     | None            | 131  | 16               | 48
alexnet_v2                     | None       | alexnet_v2/conv5     | None            | 163  | 16               | 64
alexnet_v2                     | None       | alexnet_v2/pool5     | None            | 195  | 32               | 64
alexnet_v2                     | 224        | alexnet_v2/conv1     | 0.136           | 11   | 4                | 0
alexnet_v2                     | 224        | alexnet_v2/pool1     | 0.136           | 19   | 8                | 0
alexnet_v2                     | 224        | alexnet_v2/conv2     | 0.552           | 51   | 8                | 16
alexnet_v2                     | 224        | alexnet_v2/conv3     | 0.743           | 99   | 16               | 32
alexnet_v2                     | 224        | alexnet_v2/conv4     | 1.125           | 131  | 16               | 48
alexnet_v2                     | 224        | alexnet_v2/conv5     | 1.380           | 163  | 16               | 64
alexnet_v2                     | 224        | alexnet_v2/pool5     | 1.380           | 195  | 32               | 64
alexnet_v2                     | 321        | alexnet_v2/conv1     | 0.283           | 11   | 4                | 0
alexnet_v2                     | 321        | alexnet_v2/pool1     | 0.284           | 19   | 8                | 0
alexnet_v2                     | 321        | alexnet_v2/conv2     | 1.171           | 51   | 8                | 16
alexnet_v2                     | 321        | alexnet_v2/conv3     | 1.602           | 99   | 16               | 32
alexnet_v2                     | 321        | alexnet_v2/conv4     | 2.462           | 131  | 16               | 48
alexnet_v2                     | 321        | alexnet_v2/conv5     | 3.036           | 163  | 16               | 64
alexnet_v2                     | 321        | alexnet_v2/pool5     | 3.036           | 195  | 32               | 64
vgg_a                          | None       | vgg_a/conv1/conv1_1  | None            | 3    | 1                | 1
vgg_a                          | None       | vgg_a/pool1          | None            | 4    | 2                | 1
vgg_a                          | None       | vgg_a/conv2/conv2_1  | None            | 8    | 2                | 3
vgg_a                          | None       | vgg_a/pool2          | None            | 10   | 4                | 3
vgg_a                          | None       | vgg_a/conv3/conv3_1  | None            | 18   | 4                | 7
vgg_a                          | None       | vgg_a/conv3/conv3_2  | None            | 26   | 4                | 11
vgg_a                          | None       | vgg_a/pool3          | None            | 30   | 8                | 11
vgg_a                          | None       | vgg_a/conv4/conv4_1  | None            | 46   | 8                | 19
vgg_a                          | None       | vgg_a/conv4/conv4_2  | None            | 62   | 8                | 27
vgg_a                          | None       | vgg_a/pool4          | None            | 70   | 16               | 27
vgg_a                          | None       | vgg_a/conv5/conv5_1  | None            | 102  | 16               | 43
vgg_a                          | None       | vgg_a/conv5/conv5_2  | None            | 134  | 16               | 59
vgg_a                          | None       | vgg_a/pool5          | None            | 150  | 32               | 59
vgg_a                          | 224        | vgg_a/conv1/conv1_1  | 0.177           | 3    | 1                | 1
vgg_a                          | 224        | vgg_a/pool1          | 0.180           | 4    | 2                | 1
vgg_a                          | 224        | vgg_a/conv2/conv2_1  | 2.031           | 8    | 2                | 3
vgg_a                          | 224        | vgg_a/pool2          | 2.033           | 10   | 4                | 3
vgg_a                          | 224        | vgg_a/conv3/conv3_1  | 3.883           | 18   | 4                | 7
vgg_a                          | 224        | vgg_a/conv3/conv3_2  | 7.583           | 26   | 4                | 11
vgg_a                          | 224        | vgg_a/pool3          | 7.584           | 30   | 8                | 11
vgg_a                          | 224        | vgg_a/conv4/conv4_1  | 9.434           | 46   | 8                | 19
vgg_a                          | 224        | vgg_a/conv4/conv4_2  | 13.134          | 62   | 8                | 27
vgg_a                          | 224        | vgg_a/pool4          | 13.134          | 70   | 16               | 27
vgg_a                          | 224        | vgg_a/conv5/conv5_1  | 14.059          | 102  | 16               | 43
vgg_a                          | 224        | vgg_a/conv5/conv5_2  | 14.984          | 134  | 16               | 59
vgg_a                          | 224        | vgg_a/pool5          | 14.984          | 150  | 32               | 59
vgg_a                          | 321        | vgg_a/conv1/conv1_1  | 0.363           | 3    | 1                | 1
vgg_a                          | 321        | vgg_a/pool1          | 0.369           | 4    | 2                | 1
vgg_a                          | 321        | vgg_a/conv2/conv2_1  | 4.147           | 8    | 2                | 3
vgg_a                          | 321        | vgg_a/pool2          | 4.151           | 10   | 4                | 3
vgg_a                          | 321        | vgg_a/conv3/conv3_1  | 7.927           | 18   | 4                | 7
vgg_a                          | 321        | vgg_a/conv3/conv3_2  | 15.479          | 26   | 4                | 11
vgg_a                          | 321        | vgg_a/pool3          | 15.480          | 30   | 8                | 11
vgg_a                          | 321        | vgg_a/conv4/conv4_1  | 19.256          | 46   | 8                | 19
vgg_a                          | 321        | vgg_a/conv4/conv4_2  | 26.806          | 62   | 8                | 27
vgg_a                          | 321        | vgg_a/pool4          | 26.807          | 70   | 16               | 27
vgg_a                          | 321        | vgg_a/conv5/conv5_1  | 28.695          | 102  | 16               | 43
vgg_a                          | 321        | vgg_a/conv5/conv5_2  | 30.583          | 134  | 16               | 59
vgg_a                          | 321        | vgg_a/pool5          | 30.583          | 150  | 32               | 59
vgg_16                         | None       | vgg_16/conv1/conv1_1 | None            | 3    | 1                | 1
vgg_16                         | None       | vgg_16/pool1         | None            | 6    | 2                | 2
vgg_16                         | None       | vgg_16/conv2/conv2_1 | None            | 10   | 2                | 4
vgg_16                         | None       | vgg_16/pool2         | None            | 16   | 4                | 6
vgg_16                         | None       | vgg_16/conv3/conv3_1 | None            | 24   | 4                | 10
vgg_16                         | None       | vgg_16/conv3/conv3_2 | None            | 32   | 4                | 14
vgg_16                         | None       | vgg_16/pool3         | None            | 44   | 8                | 18
vgg_16                         | None       | vgg_16/conv4/conv4_1 | None            | 60   | 8                | 26
vgg_16                         | None       | vgg_16/conv4/conv4_2 | None            | 76   | 8                | 34
vgg_16                         | None       | vgg_16/pool4         | None            | 100  | 16               | 42
vgg_16                         | None       | vgg_16/conv5/conv5_1 | None            | 132  | 16               | 58
vgg_16                         | None       | vgg_16/conv5/conv5_2 | None            | 164  | 16               | 74
vgg_16                         | None       | vgg_16/pool5         | None            | 212  | 32               | 90
vgg_16                         | 224        | vgg_16/conv1/conv1_1 | 0.177           | 3    | 1                | 1
vgg_16                         | 224        | vgg_16/pool1         | 3.882           | 6    | 2                | 2
vgg_16                         | 224        | vgg_16/conv2/conv2_1 | 5.734           | 10   | 2                | 4
vgg_16                         | 224        | vgg_16/pool2         | 9.436           | 16   | 4                | 6
vgg_16                         | 224        | vgg_16/conv3/conv3_1 | 11.287          | 24   | 4                | 10
vgg_16                         | 224        | vgg_16/conv3/conv3_2 | 14.987          | 32   | 4                | 14
vgg_16                         | 224        | vgg_16/pool3         | 18.688          | 44   | 8                | 18
vgg_16                         | 224        | vgg_16/conv4/conv4_1 | 20.538          | 60   | 8                | 26
vgg_16                         | 224        | vgg_16/conv4/conv4_2 | 24.238          | 76   | 8                | 34
vgg_16                         | 224        | vgg_16/pool4         | 27.938          | 100  | 16               | 42
vgg_16                         | 224        | vgg_16/conv5/conv5_1 | 28.863          | 132  | 16               | 58
vgg_16                         | 224        | vgg_16/conv5/conv5_2 | 29.788          | 164  | 16               | 74
vgg_16                         | 224        | vgg_16/pool5         | 30.713          | 212  | 32               | 90
vgg_16                         | 321        | vgg_16/conv1/conv1_1 | 0.363           | 3    | 1                | 1
vgg_16                         | 321        | vgg_16/pool1         | 7.973           | 6    | 2                | 2
vgg_16                         | 321        | vgg_16/conv2/conv2_1 | 11.751          | 10   | 2                | 4
vgg_16                         | 321        | vgg_16/pool2         | 19.307          | 16   | 4                | 6
vgg_16                         | 321        | vgg_16/conv3/conv3_1 | 23.084          | 24   | 4                | 10
vgg_16                         | 321        | vgg_16/conv3/conv3_2 | 30.635          | 32   | 4                | 14
vgg_16                         | 321        | vgg_16/pool3         | 38.188          | 44   | 8                | 18
vgg_16                         | 321        | vgg_16/conv4/conv4_1 | 41.964          | 60   | 8                | 26
vgg_16                         | 321        | vgg_16/conv4/conv4_2 | 49.514          | 76   | 8                | 34
vgg_16                         | 321        | vgg_16/pool4         | 57.066          | 100  | 16               | 42
vgg_16                         | 321        | vgg_16/conv5/conv5_1 | 58.954          | 132  | 16               | 58
vgg_16                         | 321        | vgg_16/conv5/conv5_2 | 60.841          | 164  | 16               | 74
vgg_16                         | 321        | vgg_16/pool5         | 62.729          | 212  | 32               | 90
inception_v2                   | None       | Conv2d_1a_7x7        | None            | 7    | 2                | None
inception_v2                   | None       | MaxPool_2a_3x3       | None            | 11   | 4                | None
inception_v2                   | None       | Conv2d_2b_1x1        | None            | 11   | 4                | None
inception_v2                   | None       | Conv2d_2c_3x3        | None            | 19   | 4                | None
inception_v2                   | None       | MaxPool_3a_3x3       | None            | 27   | 8                | None
inception_v2                   | None       | Mixed_3b             | None            | 59   | 8                | None
inception_v2                   | None       | Mixed_3c             | None            | 91   | 8                | None
inception_v2                   | None       | Mixed_4a             | None            | 123  | 16               | None
inception_v2                   | None       | Mixed_4b             | None            | 187  | 16               | None
inception_v2                   | None       | Mixed_4c             | None            | 251  | 16               | None
inception_v2                   | None       | Mixed_4d             | None            | 315  | 16               | None
inception_v2                   | None       | Mixed_4e             | None            | 379  | 16               | None
inception_v2                   | None       | Mixed_5a             | None            | 443  | 32               | None
inception_v2                   | None       | Mixed_5b             | None            | 571  | 32               | None
inception_v2                   | None       | Mixed_5c             | None            | 699  | 32               | None
inception_v2                   | 224        | Conv2d_1a_7x7        | 0.069           | 7    | 2                | 2
inception_v2                   | 224        | MaxPool_2a_3x3       | 0.071           | 11   | 4                | 2
inception_v2                   | 224        | Conv2d_2b_1x1        | 0.097           | 11   | 4                | 2
inception_v2                   | 224        | Conv2d_2c_3x3        | 0.791           | 19   | 4                | 6
inception_v2                   | 224        | MaxPool_3a_3x3       | 0.792           | 27   | 8                | 6
inception_v2                   | 224        | Mixed_3b             | 1.136           | 59   | 8                | 22
inception_v2                   | 224        | Mixed_3c             | 1.544           | 91   | 8                | 38
inception_v2                   | 224        | Mixed_4a             | 1.833           | 123  | 16               | 46
inception_v2                   | 224        | Mixed_4b             | 2.073           | 187  | 16               | 78
inception_v2                   | 224        | Mixed_4c             | 2.334           | 251  | 16               | 110
inception_v2                   | 224        | Mixed_4d             | 2.686           | 315  | 16               | 142
inception_v2                   | 224        | Mixed_4e             | 3.120           | 379  | 16               | 174
inception_v2                   | 224        | Mixed_5a             | 3.446           | 443  | 32               | 190
inception_v2                   | 224        | Mixed_5b             | 3.660           | 571  | 32               | 254
inception_v2                   | 224        | Mixed_5c             | 3.883           | 699  | 32               | 318
inception_v2                   | 321        | Conv2d_1a_7x7        | 0.142           | 7    | 2                | 3
inception_v2                   | 321        | MaxPool_2a_3x3       | 0.146           | 11   | 4                | 5
inception_v2                   | 321        | Conv2d_2b_1x1        | 0.200           | 11   | 4                | 5
inception_v2                   | 321        | Conv2d_2c_3x3        | 1.653           | 19   | 4                | 9
inception_v2                   | 321        | MaxPool_3a_3x3       | 1.656           | 27   | 8                | 13
inception_v2                   | 321        | Mixed_3b             | 2.393           | 59   | 8                | 29
inception_v2                   | 321        | Mixed_3c             | 3.268           | 91   | 8                | 45
inception_v2                   | 321        | Mixed_4a             | 3.898           | 123  | 16               | 61
inception_v2                   | 321        | Mixed_4b             | 4.438           | 187  | 16               | 93
inception_v2                   | 321        | Mixed_4c             | 5.025           | 251  | 16               | 125
inception_v2                   | 321        | Mixed_4d             | 5.817           | 315  | 16               | 157
inception_v2                   | 321        | Mixed_4e             | 6.795           | 379  | 16               | 189
inception_v2                   | 321        | Mixed_5a             | 7.545           | 443  | 32               | 221
inception_v2                   | 321        | Mixed_5b             | 8.073           | 571  | 32               | 285
inception_v2                   | 321        | Mixed_5c             | 8.626           | 699  | 32               | 349
inception_v2-no-separable-conv | None       | Conv2d_1a_7x7        | None            | 7    | 2                | None
inception_v2-no-separable-conv | None       | MaxPool_2a_3x3       | None            | 11   | 4                | None
inception_v2-no-separable-conv | None       | Conv2d_2b_1x1        | None            | 11   | 4                | None
inception_v2-no-separable-conv | None       | Conv2d_2c_3x3        | None            | 19   | 4                | None
inception_v2-no-separable-conv | None       | MaxPool_3a_3x3       | None            | 27   | 8                | None
inception_v2-no-separable-conv | None       | Mixed_3b             | None            | 59   | 8                | None
inception_v2-no-separable-conv | None       | Mixed_3c             | None            | 91   | 8                | None
inception_v2-no-separable-conv | None       | Mixed_4a             | None            | 123  | 16               | None
inception_v2-no-separable-conv | None       | Mixed_4b             | None            | 187  | 16               | None
inception_v2-no-separable-conv | None       | Mixed_4c             | None            | 251  | 16               | None
inception_v2-no-separable-conv | None       | Mixed_4d             | None            | 315  | 16               | None
inception_v2-no-separable-conv | None       | Mixed_4e             | None            | 379  | 16               | None
inception_v2-no-separable-conv | None       | Mixed_5a             | None            | 443  | 32               | None
inception_v2-no-separable-conv | None       | Mixed_5b             | None            | 571  | 32               | None
inception_v2-no-separable-conv | None       | Mixed_5c             | None            | 699  | 32               | None
inception_v2-no-separable-conv | 224        | Conv2d_1a_7x7        | 0.237           | 7    | 2                | 2
inception_v2-no-separable-conv | 224        | MaxPool_2a_3x3       | 0.239           | 11   | 4                | 2
inception_v2-no-separable-conv | 224        | Conv2d_2b_1x1        | 0.265           | 11   | 4                | 2
inception_v2-no-separable-conv | 224        | Conv2d_2c_3x3        | 0.959           | 19   | 4                | 6
inception_v2-no-separable-conv | 224        | MaxPool_3a_3x3       | 0.960           | 27   | 8                | 6
inception_v2-no-separable-conv | 224        | Mixed_3b             | 1.304           | 59   | 8                | 22
inception_v2-no-separable-conv | 224        | Mixed_3c             | 1.712           | 91   | 8                | 38
inception_v2-no-separable-conv | 224        | Mixed_4a             | 2.001           | 123  | 16               | 46
inception_v2-no-separable-conv | 224        | Mixed_4b             | 2.241           | 187  | 16               | 78
inception_v2-no-separable-conv | 224        | Mixed_4c             | 2.502           | 251  | 16               | 110
inception_v2-no-separable-conv | 224        | Mixed_4d             | 2.854           | 315  | 16               | 142
inception_v2-no-separable-conv | 224        | Mixed_4e             | 3.288           | 379  | 16               | 174
inception_v2-no-separable-conv | 224        | Mixed_5a             | 3.614           | 443  | 32               | 190
inception_v2-no-separable-conv | 224        | Mixed_5b             | 3.828           | 571  | 32               | 254
inception_v2-no-separable-conv | 224        | Mixed_5c             | 4.051           | 699  | 32               | 318
inception_v2-no-separable-conv | 321        | Conv2d_1a_7x7        | 0.489           | 7    | 2                | 3
inception_v2-no-separable-conv | 321        | MaxPool_2a_3x3       | 0.493           | 11   | 4                | 5
inception_v2-no-separable-conv | 321        | Conv2d_2b_1x1        | 0.547           | 11   | 4                | 5
inception_v2-no-separable-conv | 321        | Conv2d_2c_3x3        | 2.000           | 19   | 4                | 9
inception_v2-no-separable-conv | 321        | MaxPool_3a_3x3       | 2.003           | 27   | 8                | 13
inception_v2-no-separable-conv | 321        | Mixed_3b             | 2.740           | 59   | 8                | 29
inception_v2-no-separable-conv | 321        | Mixed_3c             | 3.615           | 91   | 8                | 45
inception_v2-no-separable-conv | 321        | Mixed_4a             | 4.246           | 123  | 16               | 61
inception_v2-no-separable-conv | 321        | Mixed_4b             | 4.785           | 187  | 16               | 93
inception_v2-no-separable-conv | 321        | Mixed_4c             | 5.373           | 251  | 16               | 125
inception_v2-no-separable-conv | 321        | Mixed_4d             | 6.164           | 315  | 16               | 157
inception_v2-no-separable-conv | 321        | Mixed_4e             | 7.142           | 379  | 16               | 189
inception_v2-no-separable-conv | 321        | Mixed_5a             | 7.892           | 443  | 32               | 221
inception_v2-no-separable-conv | 321        | Mixed_5b             | 8.421           | 571  | 32               | 285
inception_v2-no-separable-conv | 321        | Mixed_5c             | 8.973           | 699  | 32               | 349
inception_v3                   | None       | Conv2d_1a_3x3        | None            | 3    | 2                | 0
inception_v3                   | None       | Conv2d_2a_3x3        | None            | 7    | 2                | 0
inception_v3                   | None       | Conv2d_2b_3x3        | None            | 11   | 2                | 2
inception_v3                   | None       | MaxPool_3a_3x3       | None            | 15   | 4                | 2
inception_v3                   | None       | Conv2d_3b_1x1        | None            | 15   | 4                | 2
inception_v3                   | None       | Conv2d_4a_3x3        | None            | 23   | 4                | 2
inception_v3                   | None       | MaxPool_5a_3x3       | None            | 31   | 8                | 2
inception_v3                   | None       | Mixed_5b             | None            | 63   | 8                | 18
inception_v3                   | None       | Mixed_5c             | None            | 95   | 8                | 34
inception_v3                   | None       | Mixed_5d             | None            | 127  | 8                | 50
inception_v3                   | None       | Mixed_6a             | None            | 159  | 16               | 58
inception_v3                   | None       | Mixed_6b             | None            | 351  | 16               | 154
inception_v3                   | None       | Mixed_6c             | None            | 543  | 16               | 250
inception_v3                   | None       | Mixed_6d             | None            | 735  | 16               | 346
inception_v3                   | None       | Mixed_6e             | None            | 927  | 16               | 442
inception_v3                   | None       | Mixed_7a             | None            | 1055 | 32               | 490
inception_v3                   | None       | Mixed_7b             | None            | 1183 | 32               | 554
inception_v3                   | None       | Mixed_7c             | None            | 1311 | 32               | 618
inception_v3                   | 224        | Conv2d_1a_3x3        | 0.022           | 3    | 2                | 0
inception_v3                   | 224        | Conv2d_2a_3x3        | 0.241           | 7    | 2                | 0
inception_v3                   | 224        | Conv2d_2b_3x3        | 0.680           | 11   | 2                | 2
inception_v3                   | 224        | MaxPool_3a_3x3       | 0.681           | 15   | 4                | 2
inception_v3                   | 224        | Conv2d_3b_1x1        | 0.712           | 15   | 4                | 2
inception_v3                   | 224        | Conv2d_4a_3x3        | 1.460           | 23   | 4                | 2
inception_v3                   | 224        | MaxPool_5a_3x3       | 1.461           | 31   | 8                | 2
inception_v3                   | 224        | Mixed_5b             | 1.781           | 63   | 8                | 18
inception_v3                   | 224        | Mixed_5c             | 2.128           | 95   | 8                | 34
inception_v3                   | 224        | Mixed_5d             | 2.485           | 127  | 8                | 50
inception_v3                   | 224        | Mixed_6a             | 2.889           | 159  | 16               | 58
inception_v3                   | 224        | Mixed_6b             | 3.263           | 351  | 16               | 154
inception_v3                   | 224        | Mixed_6c             | 3.750           | 543  | 16               | 250
inception_v3                   | 224        | Mixed_6d             | 4.237           | 735  | 16               | 346
inception_v3                   | 224        | Mixed_6e             | 4.854           | 927  | 16               | 442
inception_v3                   | 224        | Mixed_7a             | 5.132           | 1055 | 32               | 490
inception_v3                   | 224        | Mixed_7b             | 5.385           | 1183 | 32               | 554
inception_v3                   | 224        | Mixed_7c             | 5.689           | 1311 | 32               | 618
inception_v3                   | 321        | Conv2d_1a_3x3        | 0.045           | 3    | 2                | 0
inception_v3                   | 321        | Conv2d_2a_3x3        | 0.506           | 7    | 2                | 0
inception_v3                   | 321        | Conv2d_2b_3x3        | 1.428           | 11   | 2                | 2
inception_v3                   | 321        | MaxPool_3a_3x3       | 1.431           | 15   | 4                | 2
inception_v3                   | 321        | Conv2d_3b_1x1        | 1.494           | 15   | 4                | 2
inception_v3                   | 321        | Conv2d_4a_3x3        | 3.092           | 23   | 4                | 2
inception_v3                   | 321        | MaxPool_5a_3x3       | 3.095           | 31   | 8                | 2
inception_v3                   | 321        | Mixed_5b             | 3.796           | 63   | 8                | 18
inception_v3                   | 321        | Mixed_5c             | 4.557           | 95   | 8                | 34
inception_v3                   | 321        | Mixed_5d             | 5.339           | 127  | 8                | 50
inception_v3                   | 321        | Mixed_6a             | 6.241           | 159  | 16               | 58
inception_v3                   | 321        | Mixed_6b             | 7.082           | 351  | 16               | 154
inception_v3                   | 321        | Mixed_6c             | 8.178           | 543  | 16               | 250
inception_v3                   | 321        | Mixed_6d             | 9.275           | 735  | 16               | 346
inception_v3                   | 321        | Mixed_6e             | 10.663          | 927  | 16               | 442
inception_v3                   | 321        | Mixed_7a             | 11.303          | 1055 | 32               | 490
inception_v3                   | 321        | Mixed_7b             | 11.948          | 1183 | 32               | 554
inception_v3                   | 321        | Mixed_7c             | 12.727          | 1311 | 32               | 618
inception_v4                   | None       | Conv2d_1a_3x3        | None            | 3    | 2                | 0
inception_v4                   | None       | Conv2d_2a_3x3        | None            | 7    | 2                | 0
inception_v4                   | None       | Conv2d_2b_3x3        | None            | 11   | 2                | 2
inception_v4                   | None       | Mixed_3a             | None            | 15   | 4                | 2
inception_v4                   | None       | Mixed_4a             | None            | 47   | 4                | 14
inception_v4                   | None       | Mixed_5a             | None            | 55   | 8                | 14
inception_v4                   | None       | Mixed_5b             | None            | 87   | 8                | 30
inception_v4                   | None       | Mixed_5c             | None            | 119  | 8                | 46
inception_v4                   | None       | Mixed_5d             | None            | 151  | 8                | 62
inception_v4                   | None       | Mixed_5e             | None            | 183  | 8                | 78
inception_v4                   | None       | Mixed_6a             | None            | 215  | 16               | 86
inception_v4                   | None       | Mixed_6b             | None            | 407  | 16               | 182
inception_v4                   | None       | Mixed_6c             | None            | 599  | 16               | 278
inception_v4                   | None       | Mixed_6d             | None            | 791  | 16               | 374
inception_v4                   | None       | Mixed_6e             | None            | 983  | 16               | 470
inception_v4                   | None       | Mixed_6f             | None            | 1175 | 16               | 566
inception_v4                   | None       | Mixed_6g             | None            | 1367 | 16               | 662
inception_v4                   | None       | Mixed_6h             | None            | 1559 | 16               | 758
inception_v4                   | None       | Mixed_7a             | None            | 1687 | 32               | 806
inception_v4                   | None       | Mixed_7b             | None            | 1815 | 32               | 870
inception_v4                   | None       | Mixed_7c             | None            | 1943 | 32               | 934
inception_v4                   | None       | Mixed_7d             | None            | 2071 | 32               | 998
inception_v4                   | 224        | Conv2d_1a_3x3        | 0.022           | 3    | 2                | 0
inception_v4                   | 224        | Conv2d_2a_3x3        | 0.241           | 7    | 2                | 0
inception_v4                   | 224        | Conv2d_2b_3x3        | 0.680           | 11   | 2                | 2
inception_v4                   | 224        | Mixed_3a             | 1.004           | 15   | 4                | 2
inception_v4                   | 224        | Mixed_4a             | 2.057           | 47   | 4                | 14
inception_v4                   | 224        | Mixed_5a             | 2.473           | 55   | 8                | 14
inception_v4                   | 224        | Mixed_5b             | 2.871           | 87   | 8                | 30
inception_v4                   | 224        | Mixed_5c             | 3.269           | 119  | 8                | 46
inception_v4                   | 224        | Mixed_5d             | 3.668           | 151  | 8                | 62
inception_v4                   | 224        | Mixed_5e             | 4.066           | 183  | 8                | 78
inception_v4                   | 224        | Mixed_6a             | 5.173           | 215  | 16               | 86
inception_v4                   | 224        | Mixed_6b             | 6.019           | 407  | 16               | 182
inception_v4                   | 224        | Mixed_6c             | 6.865           | 599  | 16               | 278
inception_v4                   | 224        | Mixed_6d             | 7.711           | 791  | 16               | 374
inception_v4                   | 224        | Mixed_6e             | 8.557           | 983  | 16               | 470
inception_v4                   | 224        | Mixed_6f             | 9.403           | 1175 | 16               | 566
inception_v4                   | 224        | Mixed_6g             | 10.249          | 1367 | 16               | 662
inception_v4                   | 224        | Mixed_6h             | 11.095          | 1559 | 16               | 758
inception_v4                   | 224        | Mixed_7a             | 11.588          | 1687 | 32               | 806
inception_v4                   | 224        | Mixed_7b             | 11.815          | 1815 | 32               | 870
inception_v4                   | 224        | Mixed_7c             | 12.043          | 1943 | 32               | 934
inception_v4                   | 224        | Mixed_7d             | 12.271          | 2071 | 32               | 998
inception_v4                   | 321        | Conv2d_1a_3x3        | 0.045           | 3    | 2                | 0
inception_v4                   | 321        | Conv2d_2a_3x3        | 0.506           | 7    | 2                | 0
inception_v4                   | 321        | Conv2d_2b_3x3        | 1.428           | 11   | 2                | 2
inception_v4                   | 321        | Mixed_3a             | 2.105           | 15   | 4                | 2
inception_v4                   | 321        | Mixed_4a             | 4.332           | 47   | 4                | 14
inception_v4                   | 321        | Mixed_5a             | 5.243           | 55   | 8                | 14
inception_v4                   | 321        | Mixed_5b             | 6.115           | 87   | 8                | 30
inception_v4                   | 321        | Mixed_5c             | 6.987           | 119  | 8                | 46
inception_v4                   | 321        | Mixed_5d             | 7.859           | 151  | 8                | 62
inception_v4                   | 321        | Mixed_5e             | 8.731           | 183  | 8                | 78
inception_v4                   | 321        | Mixed_6a             | 11.189          | 215  | 16               | 86
inception_v4                   | 321        | Mixed_6b             | 13.092          | 407  | 16               | 182
inception_v4                   | 321        | Mixed_6c             | 14.996          | 599  | 16               | 278
inception_v4                   | 321        | Mixed_6d             | 16.899          | 791  | 16               | 374
inception_v4                   | 321        | Mixed_6e             | 18.802          | 983  | 16               | 470
inception_v4                   | 321        | Mixed_6f             | 20.706          | 1175 | 16               | 566
inception_v4                   | 321        | Mixed_6g             | 22.609          | 1367 | 16               | 662
inception_v4                   | 321        | Mixed_6h             | 24.513          | 1559 | 16               | 758
inception_v4                   | 321        | Mixed_7a             | 25.640          | 1687 | 32               | 806
inception_v4                   | 321        | Mixed_7b             | 26.223          | 1815 | 32               | 870
inception_v4                   | 321        | Mixed_7c             | 26.807          | 1943 | 32               | 934
inception_v4                   | 321        | Mixed_7d             | 27.390          | 2071 | 32               | 998
inception_resnet_v2            | None       | Conv2d_1a_3x3        | None            | 3    | 2                | 0
inception_resnet_v2            | None       | Conv2d_2a_3x3        | None            | 7    | 2                | 0
inception_resnet_v2            | None       | Conv2d_2b_3x3        | None            | 11   | 2                | 2
inception_resnet_v2            | None       | MaxPool_3a_3x3       | None            | 15   | 4                | 2
inception_resnet_v2            | None       | Conv2d_3b_1x1        | None            | 15   | 4                | 2
inception_resnet_v2            | None       | Conv2d_4a_3x3        | None            | 23   | 4                | 2
inception_resnet_v2            | None       | MaxPool_5a_3x3       | None            | 31   | 8                | 2
inception_resnet_v2            | None       | Mixed_5b             | None            | 63   | 8                | 18
inception_resnet_v2            | None       | Mixed_6a             | None            | 415  | 16               | 186
inception_resnet_v2            | None       | PreAuxLogits         | None            | 2335 | 16               | 1146
inception_resnet_v2            | None       | Mixed_7a             | None            | 2399 | 32               | 1162
inception_resnet_v2            | None       | Conv2d_7b_1x1        | None            | 3039 | 32               | 1482
inception_resnet_v2            | 224        | Conv2d_1a_3x3        | 0.022           | 3    | 2                | 0
inception_resnet_v2            | 224        | Conv2d_2a_3x3        | 0.241           | 7    | 2                | 0
inception_resnet_v2            | 224        | Conv2d_2b_3x3        | 0.680           | 11   | 2                | 2
inception_resnet_v2            | 224        | MaxPool_3a_3x3       | 0.681           | 15   | 4                | 2
inception_resnet_v2            | 224        | Conv2d_3b_1x1        | 0.712           | 15   | 4                | 2
inception_resnet_v2            | 224        | Conv2d_4a_3x3        | 1.460           | 23   | 4                | 2
inception_resnet_v2            | 224        | MaxPool_5a_3x3       | 1.461           | 31   | 8                | 2
inception_resnet_v2            | 224        | Mixed_5b             | 1.796           | 63   | 8                | 18
inception_resnet_v2            | 224        | Mixed_6a             | 4.747           | 415  | 16               | 186
inception_resnet_v2            | 224        | PreAuxLogits         | 11.235          | 2335 | 16               | 1146
inception_resnet_v2            | 224        | Mixed_7a             | 11.786          | 2399 | 32               | 1162
inception_resnet_v2            | 224        | Conv2d_7b_1x1        | 12.963          | 3039 | 32               | 1482
inception_resnet_v2            | 321        | Conv2d_1a_3x3        | 0.045           | 3    | 2                | 0
inception_resnet_v2            | 321        | Conv2d_2a_3x3        | 0.506           | 7    | 2                | 0
inception_resnet_v2            | 321        | Conv2d_2b_3x3        | 1.428           | 11   | 2                | 2
inception_resnet_v2            | 321        | MaxPool_3a_3x3       | 1.431           | 15   | 4                | 2
inception_resnet_v2            | 321        | Conv2d_3b_1x1        | 1.494           | 15   | 4                | 2
inception_resnet_v2            | 321        | Conv2d_4a_3x3        | 3.092           | 23   | 4                | 2
inception_resnet_v2            | 321        | MaxPool_5a_3x3       | 3.095           | 31   | 8                | 2
inception_resnet_v2            | 321        | Mixed_5b             | 3.829           | 63   | 8                | 18
inception_resnet_v2            | 321        | Mixed_6a             | 10.327          | 415  | 16               | 186
inception_resnet_v2            | 321        | PreAuxLogits         | 24.924          | 2335 | 16               | 1146
inception_resnet_v2            | 321        | Mixed_7a             | 26.201          | 2399 | 32               | 1162
inception_resnet_v2            | 321        | Conv2d_7b_1x1        | 29.215          | 3039 | 32               | 1482
inception_resnet_v2-same       | None       | Conv2d_1a_3x3        | None            | 3    | 2                | None
inception_resnet_v2-same       | None       | Conv2d_2a_3x3        | None            | 7    | 2                | None
inception_resnet_v2-same       | None       | Conv2d_2b_3x3        | None            | 11   | 2                | None
inception_resnet_v2-same       | None       | MaxPool_3a_3x3       | None            | 15   | 4                | None
inception_resnet_v2-same       | None       | Conv2d_3b_1x1        | None            | 15   | 4                | None
inception_resnet_v2-same       | None       | Conv2d_4a_3x3        | None            | 23   | 4                | None
inception_resnet_v2-same       | None       | MaxPool_5a_3x3       | None            | 31   | 8                | None
inception_resnet_v2-same       | None       | Mixed_5b             | None            | 63   | 8                | None
inception_resnet_v2-same       | None       | Mixed_6a             | None            | 415  | 16               | None
inception_resnet_v2-same       | None       | PreAuxLogits         | None            | 2335 | 16               | None
inception_resnet_v2-same       | None       | Mixed_7a             | None            | 2399 | 32               | None
inception_resnet_v2-same       | None       | Conv2d_7b_1x1        | None            | 3039 | 32               | None
inception_resnet_v2-same       | 224        | Conv2d_1a_3x3        | 0.022           | 3    | 2                | 0
inception_resnet_v2-same       | 224        | Conv2d_2a_3x3        | 0.254           | 7    | 2                | 2
inception_resnet_v2-same       | 224        | Conv2d_2b_3x3        | 0.717           | 11   | 2                | 4
inception_resnet_v2-same       | 224        | MaxPool_3a_3x3       | 0.719           | 15   | 4                | 4
inception_resnet_v2-same       | 224        | Conv2d_3b_1x1        | 0.751           | 15   | 4                | 4
inception_resnet_v2-same       | 224        | Conv2d_4a_3x3        | 1.619           | 23   | 4                | 8
inception_resnet_v2-same       | 224        | MaxPool_5a_3x3       | 1.620           | 31   | 8                | 8
inception_resnet_v2-same       | 224        | Mixed_5b             | 2.041           | 63   | 8                | 24
inception_resnet_v2-same       | 224        | Mixed_6a             | 5.804           | 415  | 16               | 192
inception_resnet_v2-same       | 224        | PreAuxLogits         | 14.634          | 2335 | 16               | 1152
inception_resnet_v2-same       | 224        | Mixed_7a             | 15.456          | 2399 | 32               | 1168
inception_resnet_v2-same       | 224        | Conv2d_7b_1x1        | 17.763          | 3039 | 32               | 1488
inception_resnet_v2-same       | 321        | Conv2d_1a_3x3        | 0.046           | 3    | 2                | 1
inception_resnet_v2-same       | 321        | Conv2d_2a_3x3        | 0.524           | 7    | 2                | 3
inception_resnet_v2-same       | 321        | Conv2d_2b_3x3        | 1.481           | 11   | 2                | 5
inception_resnet_v2-same       | 321        | MaxPool_3a_3x3       | 1.485           | 15   | 4                | 7
inception_resnet_v2-same       | 321        | Conv2d_3b_1x1        | 1.553           | 15   | 4                | 7
inception_resnet_v2-same       | 321        | Conv2d_4a_3x3        | 3.368           | 23   | 4                | 11
inception_resnet_v2-same       | 321        | MaxPool_5a_3x3       | 3.371           | 31   | 8                | 15
inception_resnet_v2-same       | 321        | Mixed_5b             | 4.273           | 63   | 8                | 31
inception_resnet_v2-same       | 321        | Mixed_6a             | 12.424          | 415  | 16               | 207
inception_resnet_v2-same       | 321        | PreAuxLogits         | 32.293          | 2335 | 16               | 1167
inception_resnet_v2-same       | 321        | Mixed_7a             | 34.192          | 2399 | 32               | 1199
inception_resnet_v2-same       | 321        | Conv2d_7b_1x1        | 39.890          | 3039 | 32               | 1519
mobilenet_v1                   | None       | Conv2d_0             | None            | 3    | 2                | None
mobilenet_v1                   | None       | Conv2d_1_pointwise   | None            | 7    | 2                | None
mobilenet_v1                   | None       | Conv2d_2_pointwise   | None            | 11   | 4                | None
mobilenet_v1                   | None       | Conv2d_3_pointwise   | None            | 19   | 4                | None
mobilenet_v1                   | None       | Conv2d_4_pointwise   | None            | 27   | 8                | None
mobilenet_v1                   | None       | Conv2d_5_pointwise   | None            | 43   | 8                | None
mobilenet_v1                   | None       | Conv2d_6_pointwise   | None            | 59   | 16               | None
mobilenet_v1                   | None       | Conv2d_7_pointwise   | None            | 91   | 16               | None
mobilenet_v1                   | None       | Conv2d_8_pointwise   | None            | 123  | 16               | None
mobilenet_v1                   | None       | Conv2d_9_pointwise   | None            | 155  | 16               | None
mobilenet_v1                   | None       | Conv2d_10_pointwise  | None            | 187  | 16               | None
mobilenet_v1                   | None       | Conv2d_11_pointwise  | None            | 219  | 16               | None
mobilenet_v1                   | None       | Conv2d_12_pointwise  | None            | 251  | 32               | None
mobilenet_v1                   | None       | Conv2d_13_pointwise  | None            | 315  | 32               | None
mobilenet_v1                   | 224        | Conv2d_0             | 0.022           | 3    | 2                | 0
mobilenet_v1                   | 224        | Conv2d_1_pointwise   | 0.082           | 7    | 2                | 2
mobilenet_v1                   | 224        | Conv2d_2_pointwise   | 0.137           | 11   | 4                | 2
mobilenet_v1                   | 224        | Conv2d_3_pointwise   | 0.248           | 19   | 4                | 6
mobilenet_v1                   | 224        | Conv2d_4_pointwise   | 0.302           | 27   | 8                | 6
mobilenet_v1                   | 224        | Conv2d_5_pointwise   | 0.409           | 43   | 8                | 14
mobilenet_v1                   | 224        | Conv2d_6_pointwise   | 0.461           | 59   | 16               | 14
mobilenet_v1                   | 224        | Conv2d_7_pointwise   | 0.566           | 91   | 16               | 30
mobilenet_v1                   | 224        | Conv2d_8_pointwise   | 0.671           | 123  | 16               | 46
mobilenet_v1                   | 224        | Conv2d_9_pointwise   | 0.775           | 155  | 16               | 62
mobilenet_v1                   | 224        | Conv2d_10_pointwise  | 0.880           | 187  | 16               | 78
mobilenet_v1                   | 224        | Conv2d_11_pointwise  | 0.985           | 219  | 16               | 94
mobilenet_v1                   | 224        | Conv2d_12_pointwise  | 1.037           | 251  | 32               | 94
mobilenet_v1                   | 224        | Conv2d_13_pointwise  | 1.140           | 315  | 32               | 126
mobilenet_v1                   | 321        | Conv2d_0             | 0.046           | 3    | 2                | 1
mobilenet_v1                   | 321        | Conv2d_1_pointwise   | 0.169           | 7    | 2                | 3
mobilenet_v1                   | 321        | Conv2d_2_pointwise   | 0.286           | 11   | 4                | 5
mobilenet_v1                   | 321        | Conv2d_3_pointwise   | 0.517           | 19   | 4                | 9
mobilenet_v1                   | 321        | Conv2d_4_pointwise   | 0.632           | 27   | 8                | 13
mobilenet_v1                   | 321        | Conv2d_5_pointwise   | 0.861           | 43   | 8                | 21
mobilenet_v1                   | 321        | Conv2d_6_pointwise   | 0.979           | 59   | 16               | 29
mobilenet_v1                   | 321        | Conv2d_7_pointwise   | 1.215           | 91   | 16               | 45
mobilenet_v1                   | 321        | Conv2d_8_pointwise   | 1.450           | 123  | 16               | 61
mobilenet_v1                   | 321        | Conv2d_9_pointwise   | 1.686           | 155  | 16               | 77
mobilenet_v1                   | 321        | Conv2d_10_pointwise  | 1.922           | 187  | 16               | 93
mobilenet_v1                   | 321        | Conv2d_11_pointwise  | 2.158           | 219  | 16               | 109
mobilenet_v1                   | 321        | Conv2d_12_pointwise  | 2.286           | 251  | 32               | 125
mobilenet_v1                   | 321        | Conv2d_13_pointwise  | 2.542           | 315  | 32               | 157
mobilenet_v1_075               | None       | Conv2d_0             | None            | 3    | 2                | None
mobilenet_v1_075               | None       | Conv2d_1_pointwise   | None            | 7    | 2                | None
mobilenet_v1_075               | None       | Conv2d_2_pointwise   | None            | 11   | 4                | None
mobilenet_v1_075               | None       | Conv2d_3_pointwise   | None            | 19   | 4                | None
mobilenet_v1_075               | None       | Conv2d_4_pointwise   | None            | 27   | 8                | None
mobilenet_v1_075               | None       | Conv2d_5_pointwise   | None            | 43   | 8                | None
mobilenet_v1_075               | None       | Conv2d_6_pointwise   | None            | 59   | 16               | None
mobilenet_v1_075               | None       | Conv2d_7_pointwise   | None            | 91   | 16               | None
mobilenet_v1_075               | None       | Conv2d_8_pointwise   | None            | 123  | 16               | None
mobilenet_v1_075               | None       | Conv2d_9_pointwise   | None            | 155  | 16               | None
mobilenet_v1_075               | None       | Conv2d_10_pointwise  | None            | 187  | 16               | None
mobilenet_v1_075               | None       | Conv2d_11_pointwise  | None            | 219  | 16               | None
mobilenet_v1_075               | None       | Conv2d_12_pointwise  | None            | 251  | 32               | None
mobilenet_v1_075               | None       | Conv2d_13_pointwise  | None            | 315  | 32               | None
mobilenet_v1_075               | 224        | Conv2d_0             | 0.017           | 3    | 2                | 0
mobilenet_v1_075               | 224        | Conv2d_1_pointwise   | 0.052           | 7    | 2                | 2
mobilenet_v1_075               | 224        | Conv2d_2_pointwise   | 0.084           | 11   | 4                | 2
mobilenet_v1_075               | 224        | Conv2d_3_pointwise   | 0.148           | 19   | 4                | 6
mobilenet_v1_075               | 224        | Conv2d_4_pointwise   | 0.178           | 27   | 8                | 6
mobilenet_v1_075               | 224        | Conv2d_5_pointwise   | 0.239           | 43   | 8                | 14
mobilenet_v1_075               | 224        | Conv2d_6_pointwise   | 0.269           | 59   | 16               | 14
mobilenet_v1_075               | 224        | Conv2d_7_pointwise   | 0.328           | 91   | 16               | 30
mobilenet_v1_075               | 224        | Conv2d_8_pointwise   | 0.387           | 123  | 16               | 46
mobilenet_v1_075               | 224        | Conv2d_9_pointwise   | 0.447           | 155  | 16               | 62
mobilenet_v1_075               | 224        | Conv2d_10_pointwise  | 0.506           | 187  | 16               | 78
mobilenet_v1_075               | 224        | Conv2d_11_pointwise  | 0.565           | 219  | 16               | 94
mobilenet_v1_075               | 224        | Conv2d_12_pointwise  | 0.594           | 251  | 32               | 94
mobilenet_v1_075               | 224        | Conv2d_13_pointwise  | 0.653           | 315  | 32               | 126
mobilenet_v1_075               | 321        | Conv2d_0             | 0.034           | 3    | 2                | 1
mobilenet_v1_075               | 321        | Conv2d_1_pointwise   | 0.107           | 7    | 2                | 3
mobilenet_v1_075               | 321        | Conv2d_2_pointwise   | 0.174           | 11   | 4                | 5
mobilenet_v1_075               | 321        | Conv2d_3_pointwise   | 0.308           | 19   | 4                | 9
mobilenet_v1_075               | 321        | Conv2d_4_pointwise   | 0.373           | 27   | 8                | 13
mobilenet_v1_075               | 321        | Conv2d_5_pointwise   | 0.503           | 43   | 8                | 21
mobilenet_v1_075               | 321        | Conv2d_6_pointwise   | 0.570           | 59   | 16               | 29
mobilenet_v1_075               | 321        | Conv2d_7_pointwise   | 0.704           | 91   | 16               | 45
mobilenet_v1_075               | 321        | Conv2d_8_pointwise   | 0.837           | 123  | 16               | 61
mobilenet_v1_075               | 321        | Conv2d_9_pointwise   | 0.970           | 155  | 16               | 77
mobilenet_v1_075               | 321        | Conv2d_10_pointwise  | 1.104           | 187  | 16               | 93
mobilenet_v1_075               | 321        | Conv2d_11_pointwise  | 1.237           | 219  | 16               | 109
mobilenet_v1_075               | 321        | Conv2d_12_pointwise  | 1.310           | 251  | 32               | 125
mobilenet_v1_075               | 321        | Conv2d_13_pointwise  | 1.454           | 315  | 32               | 157
resnet_v1_50                   | None       | resnet_v1_50/block1  | None            | 35   | 8                | None
resnet_v1_50                   | None       | resnet_v1_50/block2  | None            | 99   | 16               | None
resnet_v1_50                   | None       | resnet_v1_50/block3  | None            | 291  | 32               | None
resnet_v1_50                   | None       | resnet_v1_50/block4  | None            | 483  | 32               | None
resnet_v1_50                   | 224        | resnet_v1_50/block1  | 1.325           | 35   | 8                | 15
resnet_v1_50                   | 224        | resnet_v1_50/block2  | 2.977           | 99   | 16               | 47
resnet_v1_50                   | 224        | resnet_v1_50/block3  | 5.502           | 291  | 32               | 143
resnet_v1_50                   | 224        | resnet_v1_50/block4  | 6.967           | 483  | 32               | 239
resnet_v1_50                   | 321        | resnet_v1_50/block1  | 2.771           | 35   | 8                | 17
resnet_v1_50                   | 321        | resnet_v1_50/block2  | 6.322           | 99   | 16               | 49
resnet_v1_50                   | 321        | resnet_v1_50/block3  | 12.022          | 291  | 32               | 145
resnet_v1_50                   | 321        | resnet_v1_50/block4  | 15.639          | 483  | 32               | 241
resnet_v1_101                  | None       | resnet_v1_101/block1 | None            | 35   | 8                | None
resnet_v1_101                  | None       | resnet_v1_101/block2 | None            | 99   | 16               | None
resnet_v1_101                  | None       | resnet_v1_101/block3 | None            | 835  | 32               | None
resnet_v1_101                  | None       | resnet_v1_101/block4 | None            | 1027 | 32               | None
resnet_v1_101                  | 224        | resnet_v1_101/block1 | 1.325           | 35   | 8                | 15
resnet_v1_101                  | 224        | resnet_v1_101/block2 | 2.977           | 99   | 16               | 47
resnet_v1_101                  | 224        | resnet_v1_101/block3 | 12.930          | 835  | 32               | 415
resnet_v1_101                  | 224        | resnet_v1_101/block4 | 14.395          | 1027 | 32               | 511
resnet_v1_101                  | 321        | resnet_v1_101/block1 | 2.771           | 35   | 8                | 17
resnet_v1_101                  | 321        | resnet_v1_101/block2 | 6.322           | 99   | 16               | 49
resnet_v1_101                  | 321        | resnet_v1_101/block3 | 28.734          | 835  | 32               | 417
resnet_v1_101                  | 321        | resnet_v1_101/block4 | 32.351          | 1027 | 32               | 513
resnet_v1_152                  | None       | resnet_v1_152/block1 | None            | 35   | 8                | None
resnet_v1_152                  | None       | resnet_v1_152/block2 | None            | 163  | 16               | None
resnet_v1_152                  | None       | resnet_v1_152/block3 | None            | 1315 | 32               | None
resnet_v1_152                  | None       | resnet_v1_152/block4 | None            | 1507 | 32               | None
resnet_v1_152                  | 224        | resnet_v1_152/block1 | 1.325           | 35   | 8                | 15
resnet_v1_152                  | 224        | resnet_v1_152/block2 | 4.726           | 163  | 16               | 79
resnet_v1_152                  | 224        | resnet_v1_152/block3 | 20.359          | 1315 | 32               | 655
resnet_v1_152                  | 224        | resnet_v1_152/block4 | 21.824          | 1507 | 32               | 751
resnet_v1_152                  | 321        | resnet_v1_152/block1 | 2.771           | 35   | 8                | 17
resnet_v1_152                  | 321        | resnet_v1_152/block2 | 10.071          | 163  | 16               | 81
resnet_v1_152                  | 321        | resnet_v1_152/block3 | 45.264          | 1315 | 32               | 657
resnet_v1_152                  | 321        | resnet_v1_152/block4 | 48.881          | 1507 | 32               | 753
resnet_v1_200                  | None       | resnet_v1_200/block1 | None            | 35   | 8                | None
resnet_v1_200                  | None       | resnet_v1_200/block2 | None            | 419  | 16               | None
resnet_v1_200                  | None       | resnet_v1_200/block3 | None            | 1571 | 32               | None
resnet_v1_200                  | None       | resnet_v1_200/block4 | None            | 1763 | 32               | None
resnet_v1_200                  | 224        | resnet_v1_200/block1 | 1.325           | 35   | 8                | 15
resnet_v1_200                  | 224        | resnet_v1_200/block2 | 11.720          | 419  | 16               | 207
resnet_v1_200                  | 224        | resnet_v1_200/block3 | 27.353          | 1571 | 32               | 783
resnet_v1_200                  | 224        | resnet_v1_200/block4 | 28.818          | 1763 | 32               | 879
resnet_v1_200                  | 321        | resnet_v1_200/block1 | 2.771           | 35   | 8                | 17
resnet_v1_200                  | 321        | resnet_v1_200/block2 | 25.067          | 419  | 16               | 209
resnet_v1_200                  | 321        | resnet_v1_200/block3 | 60.260          | 1571 | 32               | 785
resnet_v1_200                  | 321        | resnet_v1_200/block4 | 63.877          | 1763 | 32               | 881
resnet_v2_50                   | None       | resnet_v2_50/block1  | None            | 35   | 8                | None
resnet_v2_50                   | None       | resnet_v2_50/block2  | None            | 99   | 16               | None
resnet_v2_50                   | None       | resnet_v2_50/block3  | None            | 291  | 32               | None
resnet_v2_50                   | None       | resnet_v2_50/block4  | None            | 483  | 32               | None
resnet_v2_50                   | 224        | resnet_v2_50/block1  | 1.329           | 35   | 8                | 15
resnet_v2_50                   | 224        | resnet_v2_50/block2  | 2.982           | 99   | 16               | 47
resnet_v2_50                   | 224        | resnet_v2_50/block3  | 5.509           | 291  | 32               | 143
resnet_v2_50                   | 224        | resnet_v2_50/block4  | 6.974           | 483  | 32               | 239
resnet_v2_50                   | 321        | resnet_v2_50/block1  | 2.778           | 35   | 8                | 17
resnet_v2_50                   | 321        | resnet_v2_50/block2  | 6.333           | 99   | 16               | 49
resnet_v2_50                   | 321        | resnet_v2_50/block3  | 12.035          | 291  | 32               | 145
resnet_v2_50                   | 321        | resnet_v2_50/block4  | 15.653          | 483  | 32               | 241
resnet_v2_101                  | None       | resnet_v2_101/block1 | None            | 35   | 8                | None
resnet_v2_101                  | None       | resnet_v2_101/block2 | None            | 99   | 16               | None
resnet_v2_101                  | None       | resnet_v2_101/block3 | None            | 835  | 32               | None
resnet_v2_101                  | None       | resnet_v2_101/block4 | None            | 1027 | 32               | None
resnet_v2_101                  | 224        | resnet_v2_101/block1 | 1.329           | 35   | 8                | 15
resnet_v2_101                  | 224        | resnet_v2_101/block2 | 2.982           | 99   | 16               | 47
resnet_v2_101                  | 224        | resnet_v2_101/block3 | 12.940          | 835  | 32               | 415
resnet_v2_101                  | 224        | resnet_v2_101/block4 | 14.405          | 1027 | 32               | 511
resnet_v2_101                  | 321        | resnet_v2_101/block1 | 2.778           | 35   | 8                | 17
resnet_v2_101                  | 321        | resnet_v2_101/block2 | 6.333           | 99   | 16               | 49
resnet_v2_101                  | 321        | resnet_v2_101/block3 | 28.756          | 835  | 32               | 417
resnet_v2_101                  | 321        | resnet_v2_101/block4 | 32.374          | 1027 | 32               | 513
resnet_v2_152                  | None       | resnet_v2_152/block1 | None            | 35   | 8                | None
resnet_v2_152                  | None       | resnet_v2_152/block2 | None            | 163  | 16               | None
resnet_v2_152                  | None       | resnet_v2_152/block3 | None            | 1315 | 32               | None
resnet_v2_152                  | None       | resnet_v2_152/block4 | None            | 1507 | 32               | None
resnet_v2_152                  | 224        | resnet_v2_152/block1 | 1.329           | 35   | 8                | 15
resnet_v2_152                  | 224        | resnet_v2_152/block2 | 4.732           | 163  | 16               | 79
resnet_v2_152                  | 224        | resnet_v2_152/block3 | 20.373          | 1315 | 32               | 655
resnet_v2_152                  | 224        | resnet_v2_152/block4 | 21.838          | 1507 | 32               | 751
resnet_v2_152                  | 321        | resnet_v2_152/block1 | 2.778           | 35   | 8                | 17
resnet_v2_152                  | 321        | resnet_v2_152/block2 | 10.085          | 163  | 16               | 81
resnet_v2_152                  | 321        | resnet_v2_152/block3 | 45.294          | 1315 | 32               | 657
resnet_v2_152                  | 321        | resnet_v2_152/block4 | 48.912          | 1507 | 32               | 753
resnet_v2_200                  | None       | resnet_v2_200/block1 | None            | 35   | 8                | None
resnet_v2_200                  | None       | resnet_v2_200/block2 | None            | 419  | 16               | None
resnet_v2_200                  | None       | resnet_v2_200/block3 | None            | 1571 | 32               | None
resnet_v2_200                  | None       | resnet_v2_200/block4 | None            | 1763 | 32               | None
resnet_v2_200                  | 224        | resnet_v2_200/block1 | 1.329           | 35   | 8                | 15
resnet_v2_200                  | 224        | resnet_v2_200/block2 | 11.733          | 419  | 16               | 207
resnet_v2_200                  | 224        | resnet_v2_200/block3 | 27.373          | 1571 | 32               | 783
resnet_v2_200                  | 224        | resnet_v2_200/block4 | 28.839          | 1763 | 32               | 879
resnet_v2_200                  | 321        | resnet_v2_200/block1 | 2.778           | 35   | 8                | 17
resnet_v2_200                  | 321        | resnet_v2_200/block2 | 25.095          | 419  | 16               | 209
resnet_v2_200                  | 321        | resnet_v2_200/block3 | 60.305          | 1571 | 32               | 785
resnet_v2_200                  | 321        | resnet_v2_200/block4 | 63.922          | 1763 | 32               | 881

## FAQ

### What does a resolution of 'None' mean?

In this case, the input resolution is undefined. For most models, the receptive
field parameters can be computed even without knowing the input resolution. The
number of FLOPs cannot be computed in this case.

### For some networks, effective_padding shows as 'None' (eg, for Inception_v2 or Mobilenet_v1 when input size is not specified). Why is that?

This means that the padding for these networks depends on the input size. So,
unless we know exactly the input image dimensionality to be used, it is not
possible to determine the padding applied at the different layers. Look at the
other entries where the input size is fixed; for those cases, effective_padding
is not None.

This happens due to Tensorflow's implementation of the 'SAME' padding mode,
which may depend on the input feature map size to a given layer. For background
on this, see
[these notes from the TF documentation](https://www.tensorflow.org/versions/master/api_guides/python/nn#Notes_on_SAME_Convolution_Padding).

Also, note that in this case the program is not able to check if the network is
aligned (ie, it could be that the different paths from input to output have
receptive fields which are not consistently centered at the same position in the
input image).

So you should be aware that such networks might not be aligned -- the program
has no way of checking it when the padding cannot be determined.

### The receptive field parameters for network X seem different from what I expected... maybe your calculation is incorrect?

First, note that the results presented here are based on the tensorflow
implementations from the
[TF-Slim model library](https://github.com/tensorflow/models/tree/master/research/slim).
So, it is possible that due to some implementation details the RF parameters are
different.

One common case of confusion is the TF-Slim Resnet implementation, which applies
stride in the last residual unit of each block, instead of at the input
activations in the first residual unit of each block (which is what is described
in the Resnet paper) -- see
[this comment](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_utils.py#L30).
This makes the stride with respect to each convolution block potentially
different. In this case, though, note that a
[flag](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L150)
may be used to recover the original striding convention.

Second, it could be that we have a bug somewhere. While we include
[many tests](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/receptive_field/python/util/receptive_field_test.py)
in our library, it is always possible that we missed something. If you suspect
this is the case, please file a GitHub issue
[here](https://github.com/tensorflow/tensorflow/issues).

### The number of FLOPs for network X seem different from what I expected... maybe your calculation is incorrect?

First, note that the results presented here are based on the tensorflow
implementations from the
[TF-Slim model library](https://github.com/tensorflow/models/tree/master/research/slim).
So, it is possible that due to some implementation details the number of FLOPs
is different.

Second, one common confusion arises since some papers refer to FLOPs as the
number of Multiply-Add operations; in other words, some papers count a
Multiply-Add as one floating point operation while others count as two. Here, we
follow the `tensorflow.profiler` convention and count a Multiply-Add as two
operations. One noticeable counter-example is the
[ResNet paper](https://arxiv.org/abs/1512.03385), where the FLOPs mentioned in
Table 1 therein actually mean the number of Multiply-Add's (see Section 3.3 in
their paper). So there is roughly a factor of two between their paper and our
numbers.

Finally, we rely on `tensorflow.profiler` for estimating the number of floating
point operations. It could be that they have a bug somewhere, or that we are
using their library incorrectly, or that we simply have a bug somewhere else. If
you suspect this is the case, please file a GitHub issue
[here](https://github.com/tensorflow/tensorflow/issues).
