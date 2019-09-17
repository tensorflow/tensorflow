
## XMOS Microlite Integration Fork
* __This README should not be part of any eventual merge with mainline Tensorflow__
* If you are here looking for general purpose TF go to the real one (see below)
* The scope of this repo is to stage an XS3 port of the TF microlite interpreter
* If you are unsure what you are doing here please contact andrewc@xmos.com

To get started run the following command from *this* directory:

    make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET="xcore" test -j3

*Note: the -j argument can be > 3, but should be at least 3 so you can get all but one test done quickly (the other test will just take forever, sorry)*

For a primer on how to do the porting, please check out the README here: https://github.com/xmos/tensorflow/tree/master/tensorflow/lite/experimental/micro

Another, more approachable explainer on porting: https://www.oreilly.com/library/view/tinyml/9781492052036/ch04.html

| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |
[TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform
for machine learning. It has a comprehensive, flexible ecosystem of
[tools](https://www.tensorflow.org/resources/tools),
[libraries](https://www.tensorflow.org/resources/libraries-extensions), and
[community](https://www.tensorflow.org/community) resources that lets
researchers push the state-of-the-art in ML and developers easily build and
deploy ML powered applications.

TensorFlow was originally developed by researchers and engineers working on the
Google Brain team within Google's Machine Intelligence Research organization for
the purposes of conducting machine learning and deep neural networks research.
The system is general enough to be applicable in a wide variety of other
domains, as well.

TensorFlow provides stable [Python](https://www.tensorflow.org/api_docs/python)
and [C++](https://www.tensorflow.org/api_docs/cc) APIs, as well as
non-guaranteed backwards compatible API for
[other languages](https://www.tensorflow.org/api_docs).

## License

[Apache License 2.0](LICENSE)
