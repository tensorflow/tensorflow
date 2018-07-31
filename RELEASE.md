# Release 1.9.0

## Major Features And Improvements
* Updated docs for `tf.keras`: New Keras-based [get started](http://tensorflow.org/versions/r1.9/get_started),
  and [programmers guide page](http://tensorflow.org/versions/r1.9/programmers_guide/keras).
* Update `tf.keras` to the Keras 2.1.6 API.
* Added [`tf.keras.layers.CuDNNGRU`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/keras/layers/CuDNNGRU) and [`tf.keras.layers.CuDNNLSTM`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/keras/layers/CuDNNLSTM) layers. [Try it](https://colab.sandbox.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb?linkId=53292082).
* Adding support of core [feature columns](https://www.tensorflow.org/get_started/feature_columns) and [losses](https://www.tensorflow.org/api_docs/python/tf/losses) to [gradient boosted trees estimators](https://github.com/tensorflow/models/tree/master/official/boosted_trees).
* The [python interface](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/contrib/lite)
  for the [TFLite Optimizing Converter](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/README.md)
  has been expanded, and the command line interface (AKA: `toco`, `tflite_convert`) is once again
  included in the standard `pip` installation.
* Improved data-loading and text processing with:
    * [`tf.decode_compressed`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/decode_compressed)
    * [`tf.string_strip`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/string_strip)
    * [`tf.strings.regex_full_match`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/strings/regex_full_match)
* Added experimental support for new pre-made Estimators:
  * [`tf.contrib.estimator.BaselineEstimator`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/contrib/estimator/BaselineEstimator)
  * [`tf.contrib.estimator.RNNClassifier`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/contrib/estimator/RNNEstimator)
  * [`tf.contrib.estimator.RNNEstimator`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/contrib/estimator/RNNClassifier)
* The [distributions.Bijector](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/contrib/distributions/bijectors/Bijector)
  API supports broadcasting for Bijectors with new API changes.
  
## Breaking Changes
  * If you're opening empty variable scopes; replace `variable_scope('', ...)` by
    `variable_scope(tf.get_variable_scope(), ...)`.
  * Headers used for building custom ops have been moved from site-packages/external into site-packages/tensorflow/include/external.

## Bug Fixes and Other Changes

* `tfe.Network` is deprecated. Please inherit from `tf.keras.Model`.
* Layered variable names have changed in the following conditions:
  * Using `tf.keras.layers` with custom variable scopes.
  * Using `tf.layers` in  a subclassed `tf.keras.Model` class. See
    [here](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/layers) for more details
* `tf.data`:
  * `Dataset.from_generator()` now accepts an `args` list, in order to create nested generators.
  * `Dataset.list_files()` now produces determinstic results when `shuffle=False` or a `seed` is passed.
  * `tf.contrib.data.sample_from_datasets()` and `tf.contrib.data.choose_from_datasets()` make it easier to sample or deterministically choose elements from multiple datasets.
  * `tf.contrib.data.make_csv_dataset()` now supports line breaks in quoted strings, and two infrequently used arguments removed.
  * (C++) `DatasetBase::DebugString()` is now `const`.
  * (C++) `DatasetBase::MakeIterator()` has been renamed to `DatasetBase::MakeIteratorInternal()`.
  * (C++) `IteratorBase::Initialize()` method was added to support raising errors during iterator construction.
* Eager Execution:
  * Added the ability to pause recording operations for gradient computation via `tf.GradientTape.stop_recording`.
  * Updated documentation, introductory notebooks.
* `tf.keras`:
  * Move Keras code out of _impl folder and remove API files.
  * `tf.keras.Model.save_weights` now saves in TensorFlow format by default.
  * Enable dataset iterators to be passed to `tf.keras.Model` training/eval methods.
* TensorFlow Debugger (tfdbg) CLI: fix an issue in which the TensorBoard Debugger Plugin could not handle total source file size exceeding gRPC message size limit (4 MB).
* `tf.contrib`:
  * `tf.contrib.framework.zero_initializer` supports ResourceVariable.
  * Adding "constrained_optimization" to tensorflow/contrib.
* Other:
  * Add GCS Configuration Ops.
  * Changing signature of `MakeIterator` to enable propagating error status.
  * KL divergence for two Dirichlet distributions.
  * More consistent GcsFileSystem behavior for certain reads past EOF.
  * Update benchmark for tf.scan to match ranges across eager and graph modes.
  * Fixed bug in `tf.reduce_prod gradient` for complex dtypes.
  * Allow the use of '.' in variables (e.g. "hparams.parse('a.b=1.0')"), which would previously raise an error. This will correspond to an attribute name with an embedded '.' symbol (e.g. 'a.b'), which can only be accessed indirectly (e.g. through getattr and setattr).  To set this up the user will first need to explicitly add the variable to the hparam object (e.g. "hparams.add_hparam(name='a.b', value=0.0)").
  * Benchmark for tf.scan in graph and eager modes.
  * Added complex128 support to FFT, FFT2D, FFT3D, IFFT, IFFT2D, and IFFT3D.
  * Making ids unique in `nn.embedding_lookup_sparse`. This helps to reduce RPC calls for looking up the embeddings when there are repeated ids in the batch.
  * Support indicator column in boosted trees.
  * Prevent `tf.gradients()` from backpropagating through integer tensors.
  * LinearOperator[1D,2D,3D]Circulant added to `tensorflow.linalg`.
  * Conv3D, Conv3DBackpropInput, Conv3DBackpropFilter now supports arbitrary.
  * Added `tf.train.Checkpoint` for reading/writing object-based checkpoints.
  * Added LinearOperatorKronecker, a dense-free implementation of the Kronecker Product.
  * Allow LinearOperator to broadcast.
  * SavedModelBuilder will now deduplicate asset names that point to files with the same basename and the same contents. Note that this may result in new asset files included in SavedModels in cases where assets with the same name but different contents were previously overwriting each other.


## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Abdullah Alrasheed, Achal Shah, Ad-530, ADiegoCAlonso, Aditya Yogi, Ag Ramesh, akindyakov, Andy Kernahan, Anya Petrova, Aurelien Geron, Ben, Ben Barsdell, Bhavani-Subramanian, braincodercn, Brett Koonce, Brian Nemsick, Brian Zier, Bryan Heden, candy.dc, cclauss, Clayne Robison, ctiijima, Dalmo Cirne, David Norman, David T.H. Kao, DosLin, ekelsen, Elson Rodriguez, Erik Smistad, Felix Abecassis, Fergal Cotter, fo40225, foo0x29a, Freedom" Koan-Sin Tan, FréDéRic Branchaud-Charron, gdh1995, Geoffrey Irving, Giuseppe, gracehoney, Guido Zuidhof, Guillaume Klein, Guozhong Zhuang, Haggai, Harald Husum, imsheridan, Ivan Zhang, Jan Zikes, Jayaram Bobba, Jesse Benson, Jesse Gumz, Jiajia Li, Jie, jinghuangintel, Jingwen, jjsjann123, Joe Yearsley, Joel Hestness, Joel Shor, josephyearsley, Junpeng Lao, Karol M. Langner, Kb Sriram, krantideep95, Krish Ravindranath, Letian Feng, Loo Rong Jie, Lukas Geiger, Maciej, Mahmoud Abuzaina, ManHyuk, Mark Ryan, mbhuiyan, Michal Turek, Mostafa Alaa, Myungsung Kwak, Nand Dalal, Nehal J Wani, Neil Tenenholtz, ngc92, Nicholas Nadeau, P.Eng., Avs, Niranjan Hasabnis, P-Hidringer, Paul Van Eck, Peng Yu, Qing Zhao, Qingying Chen, Quanlong, Rajendra Arora, Rholais Lii, rmanyari, Robin Richtsfeld, Russell Klopfer, Sagi, Sam Sendelbach, Sandeep N Gupta, Sandip Giri, Sarah Edkins, Scott Tseng, Sdalbsoo, Sergii Khomenko, Seungwoo Choi (Biggie), Seyed Majid Azimi, Shaoning Zeng, shengfuintel, Siu Kei, Muk, Smit Shilu, soonson, Stefan Schweter, Sukhwan Kim, Sunitha Kambhampati, Taehoon Lee, tamimaddari82, Tang, Wenyi, Ted Chang, u2takey, Utkarsh Upadhyay, Vadim Markovtsev, voegtlel, Wai Hon Law, wangsiyu, Wenhao Hu, wenhao.hu, William D. Irons, Yan Facai (颜发才), Yanbo Liang, Yihong Wang, Yilei (Dolee) Yang, Yong Tang, Yuan (Terry) Tang

# Release 1.8.0

## Major Features And Improvements
* Can now pass `tf.contrib.distribute.MirroredStrategy()` to `tf.estimator.RunConfig()` to run an Estimator model on multiple GPUs on one machine.
* Add `tf.contrib.data.prefetch_to_device()`, which supports prefetching to GPU memory.
* Added Gradient Boosted Trees as pre-made Estimators: BoostedTreesClassifier, BoostedTreesRegressor.
* Add 3rd generation pipeline config for Cloud TPUs which improves performance and usability.
* `tf.contrib.bayesflow` is moving out to it's own repo.
* Added `tf.contrib.{proto,rpc}` to allow generic proto parsing and RPC communication<sup>[1](#rpc-issue)</sup>.

## Bug Fixes and Other Changes
* `tf.data`:
  * Add `tf.contrib.data.prefetch_to_device`, which enables prefetching dataset elements to GPU memory.
  * Add `tf.contrib.data.AUTOTUNE`, which allows the tf.data runtime to automatically tune the prefetch buffer sizes based on your system and environment.
  * Add `tf.contrib.data.make_csv_dataset` for building datasets of CSV files.
* Eager Execution:
  * With eager execution Datasets can now be used as standard python iterators (`for batch in dataset:`). Both `Dataset.__iter__()` and `Dataset.make_one_shot_iterator()` can now be used to create iterators when eager execution is enabled.
  * Automatic device placement has been enabled (i.e., use a GPU if available automatically, without requiring an explicit `with tf.device(“/gpu:0”)`) (Fixes #14133)
  * `tf.GradientTape` has moved out of contrib.
* `tf.keras`:
  * Added the fashion mnist dataset.
  * New data preprocessing functions: `image/random_brightness`, `sequence/TimeseriesGenerator`, and `text/hashing_trick`.
* Accelerated Linear Algebra (XLA):
  * Select and scatter in reference util and evaluator now use lexicographical order to break ties.
* TensorFlow Debugger (tfdbg) CLI:
  * During tensor-filter operations, allow exclusion of nodes by regular expressions.
  * Fix spurious background colors in some text terminals.
* `tf.contrib`:
  * Add meta-distribution BatchReshape which reshapes batch dimensions.
  * `tf.contrib.layers.recompute_grad` works for explicit gradient checkpointing on TPU.
  * Add `tf.contrib.framework.argsort`.
  * Allow `DNNBoostedTreeCombinedEstimator` to work with core versions of feature columns and losses.
  * Add non-linear image warping ops: `tf.contrib.image.sparse_image_warp`, `tf.contrib.image.dense_image_warp`, and `tf.contrib.image.interpolate_spline`.
  * Fix bug in `tf.contrib.opt.MultitaskOptimizerWrapper` where types of tensors were mismatched.
* Other:
  * Low-level graph construction now calls the TensorFlow C API. This change should be invisible to most users, but can be disabled by setting the environment variable `TF_C_API_GRAPH_CONSTRUCTION=0` in this release. Future releases will remove the ability to disable this change. Please [file a bug](https://github.com/tensorflow/tensorflow/issues/new) if you find yourself using this escape hatch.
  * Add description of shapes and a pointer to tutorial notebook in `tf.distributions.Distribution`.
  * Update scatter operations:
    * Add `tf.scatter_min` and `tf.scatter_max`
    * Extend scatter operations to work with a scalar update parameter.
  * Move cuDNN RNN ops to core for use in TensorFlow codebase only.
  * Add `float64` support for `Conv2d`, `Conv2dBackpropInput`, and `Conv2dBackpropFilter`.
  * Add `float64` support for `AvgPool`/`AvgPoolGrad`.
  * Make graph name scope thread local so that they work correctly in multi-threaded environments.
  * Update nsync synchronization library to avoid slow primitives on Linux.
  * Removed need to put nsync/public on C include path when building custom ops.
  * Add `tf.image.psnr`, `tf.image.ssim`, `tf.image.ssim_multiscale`, `tf.image.image_gradients`, `tf.image.sobel_edges`.
  * Add links to https://js.tensorflow.org.
  * Fix non-uniformity of orthogonal matrices.
  * Fix bug where multi-image Estimator eval summaries were not displayed correctly.

<a name="rpc-issue"><sup>1</sup></a> The cancellation logic of the RPC op contains a concurrency error. A fix has been submitted to master and will be part of the next release.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

4d55397500, Aghasy, Alan Du, Alan Lee, Alan Yee, Alex Wiltschko, Animesh Karnewar, Ankit Gupta, Anton Matosov, Aris L, Ben Barsdell, Brent Yi, Brett Koonce, Carl Thomé, cbockman, Chikanaga Tomoyuki, Chris Tava, CéDric Deltheil, Dahan Gong, Dalmo Cirne, Daniel Erenrich, David Norman, DavidNorman, Edd Wilder-James, Fanjin Zeng, Felix Abecassis, fo40225, George Sterpu, Giovanni Terlingen, Gor Baghdasaryan, Guillaume Klein, Hanchen Li, Ilya Polenov, Jakub Kolodziejczyk, Jason Sadler, Jayaram Bobba, Jerry Liu, jinghuangintel, Jiongyan Zhang (张炯衍), Joel Shor, Jong Wook Kim, Julian Eisenschlos, Karl Lessard, Krish Ravindranath, Loo Rong Jie, Lukas Geiger, Luke Iwanski, Mahmoud Abuzaina, ManHyuk, Marvin Richter, Maximilian Mitchell, Mohammad Ashraf Bhuiyan, msofka, Mustafa Kasap, Nathan Burnham, Nathan Luehr, Naveen Marri, ngc92, nio1814, Oleg Zabluda, Ou Changkun, Panos Ipeirotis, Paul Van Eck, Peter Lee, Piotr Czapla, qjivy, Rholais Lii, Rodrigo Formigone, Russell Klopfer, ryantimjohn, Sang Han, SebastiáN RamíRez, shengfuintel, Siby Jose Plathottam, Silver Chan, Stanislaw Antol, Taehoon Lee, Tarang Chugh, Ted Chang, Thomas Bastiani, Xian Xu, Xiaoming (Jason) Cui, Yan Facai (颜发才), yaox12, Yashal Shakti Kanungo, Yong Tang, Yuan (Terry) Tang, Yuxin Wu, Ziyue(Louis) Lu

# Release 1.7.0

## Major Features And Improvements
* Eager mode is moving out of contrib, try `tf.enable_eager_execution()`.
* Graph rewrites emulating fixed-point quantization compatible with TensorFlow Lite, supported by new `tf.contrib.quantize` package.
* Easily customize gradient computation with `tf.custom_gradient`.
* [TensorBoard Debugger Plugin](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/debugger/README.md), the graphical user interface (GUI) of TensorFlow Debugger (tfdbg), is now in alpha.
* Experimental support for reading a sqlite database as a `Dataset` with new `tf.contrib.data.SqlDataset`.
* Distributed Mutex / CriticalSection added to `tf.contrib.framework.CriticalSection`.
* Better text processing with `tf.regex_replace`.
* Easy, efficient sequence input with `tf.contrib.data.bucket_by_sequence_length`
* Initial support for `tf.contrib.tensorrt` that enables native TensorRT in
  TensorFlow.

## Bug Fixes and Other Changes
* Accelerated Linear Algebra (XLA):
  * Add `MaxPoolGradGrad` support for XLA
  * CSE pass from Tensorflow is now disabled in XLA.
* `tf.data`:
  * `tf.data.Dataset`
    * Add support for building C++ Dataset op kernels as external libraries, using the `tf.load_op_library()` mechanism.
    * `Dataset.list_files()` now shuffles its output by default.
    * `Dataset.shuffle(..., seed=tf.constant(0, dtype=tf.int64))` now yields the same sequence of elements as `Dataset.shuffle(..., seed=0)`.
  * Add `num_parallel_reads` argument to `tf.data.TFRecordDataset`.
* `tf.contrib`:
  * `tf.contrib.bayesflow.halton_sequence` now supports randomization.
  * Add support for scalars in `tf.contrib.all_reduce`.
  * Add `effective_sample_size` to `tf.contrib.bayesflow.mcmc_diagnostics`.
  * Add `potential_scale_reduction` to `tf.contrib.bayesflow.mcmc_diagnostics`.
  * Add `BatchNormalization`, `Kumaraswamy` bijectors.
  * Deprecate `tf.contrib.learn`. Please check contrib/learn/README.md for instructions on how to convert existing code.
  * `tf.contrib.data`
    * Remove deprecated `tf.contrib.data.Dataset`, `tf.contrib.data.Iterator`, `tf.contrib.data.FixedLengthRecordDataset`, `tf.contrib.data.TextLineDataset`, and `tf.contrib.data.TFRecordDataset` classes.
    * Added `bucket_by_sequence_length`, `sliding_window_batch`, and `make_batched_features_dataset`
  * Remove unmaintained `tf.contrib.ndlstm`. You can find it externally at https://github.com/tmbarchive/tfndlstm.
  * Moved most of `tf.contrib.bayesflow` to its own repo: `tfp`
* Other:
  * tf.py_func now reports the full stack trace if an exception occurs.
  * Integrate `TPUClusterResolver` with GKE's integration for Cloud TPUs.
  * Add a library for statistical testing of samplers.
  * Add Helpers to stream data from the GCE VM to a Cloud TPU.
  * Integrate ClusterResolvers with TPUEstimator.
  * Unify metropolis_hastings interface with HMC kernel.
  * Move LIBXSMM convolutions to a separate --define flag so that they are disabled by default.
  * Fix `MomentumOptimizer` lambda.
  * Reduce `tfp.layers` boilerplate via programmable docstrings.
  * Add `auc_with_confidence_intervals`, a method for computing the AUC and confidence interval with linearithmic time complexity.
  * `regression_head` now accepts customized link function, to satisfy the usage that user can define their own link function if the `array_ops.identity` does not meet the requirement.
  * Fix `initialized_value` and `initial_value` behaviors for `ResourceVariables` created from `VariableDef` protos.
  * Add TensorSpec to represent the specification of Tensors.
  * Constant folding pass is now deterministic.
  * Support `float16` `dtype` in `tf.linalg.*`.
  * Add `tf.estimator.export.TensorServingInputReceiver` that allows `tf.estimator.Estimator.export_savedmodel` to pass raw tensors to model functions.

## Deprecations

* TensorFlow 1.7 may be the last time we support Cuda versions below 8.0.
  Starting with TensorFlow 1.8 release, 8.0 will be the minimum supported
  version.
* TensorFlow 1.7 may be the last time we support cuDNN versions below 6.0.
  Starting with TensorFlow 1.8 release, 6.0 will be the minimum supported
  version.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

4d55397500, Abe, Alistair Low, Andy Kernahan, Appledore, Ben, Ben Barsdell, Boris Pfahringer, Brad Wannow, Brett Koonce, Carl Thomé, cclauss, Chengzhi Chen, Chris Drake, Christopher Yeh, Clayne Robison, Codrut Grosu, Daniel Trebbien, Danny Goodman, David Goodwin, David Norman, Deron Eriksson, Donggeon Lim, Donny Viszneki, DosLin, DylanDmitri, Francisco Guerrero, Fred Reiss, gdh1995, Giuseppe, Glenn Weidner, gracehoney, Guozhong Zhuang, Haichen "Hc" Li, Harald Husum, harumitsu.nobuta, Henry Spivey, hsm207, Jekyll Song, Jerome, Jiongyan Zhang, jjsjann123, John Sungjin Park, Johnson145, JoshVarty, Julian Wolff, Jun Wang, June-One, Kamil Sindi, Kb Sriram, Kdavis-Mozilla, Kenji, lazypanda1, Liang-Chi Hsieh, Loo Rong Jie, Mahesh Bhosale, MandarJKulkarni, ManHyuk, Marcus Ong, Marshal Hayes, Martin Pool, matthieudelaro, mdfaijul, mholzel, Michael Zhou, Ming Li, Minmin Sun, Myungjoo Ham, MyungsungKwak, Naman Kamra, Peng Yu, Penghao Cen, Phil, Raghuraman-K, resec, Rohin Mohanadas, Sandeep N Gupta, Scott Tseng, seaotterman, Seo Sanghyeon, Sergei Lebedev, Ted Chang, terrytangyuan, Tim H, tkunic, Tod, vihanjain, Yan Facai (颜发才), Yin Li, Yong Tang, Yukun Chen, Yusuke Yamada



# Release 1.6.0

## Breaking Changes
* Prebuilt binaries are now built against CUDA 9.0 and cuDNN 7.
* Prebuilt binaries will use AVX instructions. This may break TF on older CPUs.

## Major Features And Improvements
* New Optimizer internal API for non-slot variables. Descendants of AdamOptimizer that access _beta[12]_power will need to be updated.
* `tf.estimator.{FinalExporter,LatestExporter}` now export stripped SavedModels. This improves forward compatibility of the SavedModel.
* FFT support added to XLA CPU/GPU.

## Bug Fixes and Other Changes
* Documentation updates:
  * Added a second version of Getting Started, which is aimed at ML
newcomers.
  * Clarified documentation on `resize_images.align_corners` parameter.
  * Additional documentation for TPUs.
* Google Cloud Storage (GCS):
  * Add client-side throttle.
  * Add a `FlushCaches()` method to the FileSystem interface, with an implementation for GcsFileSystem.
* Other:
  * Add `tf.contrib.distributions.Kumaraswamy`.
  * `RetryingFileSystem::FlushCaches()` calls the base FileSystem's `FlushCaches()`.
  * Add `auto_correlation` to distributions.
  * Add `tf.contrib.distributions.Autoregressive`.
  * Add SeparableConv1D layer.
  * Add convolutional Flipout layers.
  * When both inputs of `tf.matmul` are bfloat16, it returns bfloat16, instead of float32.
  * Added `tf.contrib.image.connected_components`.
  * Add `tf.contrib.framework.CriticalSection` that allows atomic variable access.
  * Output variance over trees predictions for classifications tasks.
  * For `pt` and `eval` commands, allow writing tensor values to filesystem as numpy files.
  * gRPC: Propagate truncated errors (instead of returning gRPC internal error).
  * Augment `parallel_interleave` to support 2 kinds of prefetching.
  * Improved XLA support for C64-related ops log, pow, atan2, tanh.
  * Add probabilistic convolutional layers.

## API Changes
* Introducing `prepare_variance` boolean with default setting to False for backward compatibility.
* Move `layers_dense_variational_impl.py` to `layers_dense_variational.py`.

## Known Bugs
* Using XLA:GPU with CUDA 9 and CUDA 9.1 results in garbage results and/or
  `CUDA_ILLEGAL_ADDRESS` failures.

  Google discovered in mid-December 2017 that the PTX-to-SASS compiler in CUDA 9
  and CUDA 9.1 sometimes does not properly compute the carry bit when
  decomposing 64-bit address calculations with large offsets (e.g. `load [x +
  large_constant]`) into 32-bit arithmetic in SASS.

  As a result, these versions of `ptxas` miscompile most XLA programs which use
  more than 4GB of temp memory.  This results in garbage results and/or
  `CUDA_ERROR_ILLEGAL_ADDRESS` failures.

  A fix in CUDA 9.1.121 is expected in late February 2018.  We do not expect a
  fix for CUDA 9.0.x.  Until the fix is available, the only workaround is to
  [downgrade](https://developer.nvidia.com/cuda-toolkit-archive) to CUDA 8.0.x
  or disable XLA:GPU.

  TensorFlow will print a warning if you use XLA:GPU with a known-bad version of
  CUDA; see e00ba24c4038e7644da417ddc639169b6ea59122.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

4d55397500, Ag Ramesh, Aiden Scandella, Akimasa Kimura, Alex Rothberg, Allen Goodman,
amilioto, Andrei Costinescu, Andrei Nigmatulin, Anjum Sayed, Anthony Platanios,
Anush Elangovan, Armando Fandango, Ashish Kumar Ram, Ashwini Shukla, Ben, Bhavani Subramanian,
Brett Koonce, Carl Thomé, cclauss, Cesc, Changming Sun, Christoph Boeddeker, Clayne Robison,
Clemens Schulz, Clint (Woonhyuk Baek), codrut3, Cole Gerdemann, Colin Raffel, Daniel Trebbien,
Daniel Ylitalo, Daniel Zhang, Daniyar, Darjan Salaj, Dave Maclachlan, David Norman, Dong--Jian,
dongsamb, dssgsra, Edward H, eladweiss, elilienstein, Eric Lilienstein, error.d, Eunji Jeong, fanlu,
Florian Courtial, fo40225, Fred, Gregg Helt, Guozhong Zhuang, Hanchen Li, hsm207, hyunyoung2,
ImSheridan, Ishant Mrinal Haloi, Jacky Ko, Jay Young, Jean Flaherty, Jerome, JerrikEph, Jesse
Kinkead, jfaath, Jian Lin, jinghuangintel, Jiongyan Zhang, Joel Hestness, Joel Shor, Johnny Chan,
Julian Niedermeier, Julian Wolff, JxKing, K-W-W, Karl Lessard, Kasper Marstal, Keiji Ariyama,
Koan-Sin Tan, Loki Der Quaeler, Loo Rong Jie, Luke Schaefer, Lynn Jackson, ManHyuk, Matt Basta,
Matt Smith, Matthew Schulkind, Michael, michaelkhan3, Miguel Piedrafita, Mikalai Drabovich,
Mike Knapp, mjwen, mktozk, Mohamed Aly, Mohammad Ashraf Bhuiyan, Myungjoo Ham, Naman Bhalla,
Namrata-Ibm, Nathan Luehr, nathansilberman, Netzeband, Niranjan Hasabnis, Omar Aflak, Ozge
Yalcinkaya, Parth P Panchal, patrickzzy, Patryk Chrabaszcz, Paul Van Eck, Paweł Kapica, Peng Yu,
Philip Yang, Pierre Blondeau, Po-Hsien Chu, powderluv, Puyu Wang, Rajendra Arora, Rasmus, Renat
Idrisov, resec, Robin Richtsfeld, Ronald Eddy Jr, Sahil Singh, Sam Matzek, Sami Kama, sandipmgiri,
Santiago Castro, Sayed Hadi Hashemi, Scott Tseng, Sergii Khomenko, Shahid, Shengpeng Liu, Shreyash
Sharma, Shrinidhi Kl, Simone Cirillo, simsicon, Stanislav Levental, starsblinking, Stephen Lumenta,
Steven Hickson, Su Tang, Taehoon Lee, Takuya Wakisaka, Ted Chang, Ted Ying, Tijmen Verhulsdonck,
Timofey Kondrashov, vade, vaibhav, Valentin Khrulkov, vchigrin, Victor Costan, Viraj Navkal,
Vivek Rane, wagonhelm, Yan Facai (颜发才), Yanbo Liang, Yaroslav Bulatov, yegord, Yong Tang,
Yoni Tsafir, yordun, Yuan (Terry) Tang, Yuxin Wu, zhengdi, Zhengsheng Wei, 田传武

# Release 1.5.0

## Breaking Changes
* Prebuilt binaries are now built against CUDA 9.0 and cuDNN 7.
* Starting from 1.6 release, our prebuilt binaries will use AVX instructions.
  This may break TF on older CPUs.

## Major Features And Improvements
* [Eager execution](https://github.com/tensorflow/tensorflow/tree/r1.5/tensorflow/contrib/eager)
  preview version is now available.
* [TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/r1.5/tensorflow/contrib/lite)
  dev preview is now available.
* CUDA 9.0 and cuDNN 7 support.
* Accelerated Linear Algebra (XLA):
  * Add `complex64` support to XLA compiler.
  * `bfloat` support is now added to XLA infrastructure.
  * Make `ClusterSpec` propagation work with XLA devices.
  * Use a deterministic executor to generate XLA graph.
* `tf.contrib`:
  * `tf.contrib.distributions`:
    * Add `tf.contrib.distributions.Autoregressive`.
    * Make `tf.contrib.distributions` QuadratureCompound classes support batch
    * Infer `tf.contrib.distributions.RelaxedOneHotCategorical` `dtype` from arguments.
    * Make `tf.contrib.distributions` quadrature family parameterized by
      `quadrature_grid_and_prob` vs `quadrature_degree`.
    * `auto_correlation` added to `tf.contrib.distributions`
  * Add `tf.contrib.bayesflow.layers`, a collection of probabilistic (neural) layers.
  * Add `tf.contrib.bayesflow.halton_sequence`.
  * Add `tf.contrib.data.make_saveable_from_iterator.`
  * Add `tf.contrib.data.shuffle_and_repeat`.
  * Add new custom transformation: `tf.contrib.data.scan()`.
  * `tf.contrib.distributions.bijectors`:
    * Add `tf.contrib.distributions.bijectors.MaskedAutoregressiveFlow`.
    * Add `tf.contrib.distributions.bijectors.Permute`.
    * Add `tf.contrib.distributions.bijectors.Gumbel`.
    * Add `tf.contrib.distributions.bijectors.Reshape`.
    * Support shape inference (i.e., shapes containing -1) in the Reshape bijector.
* Add `streaming_precision_recall_at_equal_thresholds,` a method for computing
  streaming precision and recall with `O(num_thresholds + size of predictions)`
  time and space complexity.
* Change `RunConfig` default behavior to not set a random seed, making random
  behavior independently random on distributed workers. We expect this to
  generally improve training performance. Models that do rely on determinism
  should set a random seed explicitly.
* Replaced the implementation of `tf.flags` with `absl.flags`.
* Add support for `CUBLAS_TENSOR_OP_MATH` in fp16 GEMM
* Add support for CUDA on NVIDIA Tegra devices

## Bug Fixes and Other Changes
* Documentation updates:
  * Clarified that you can only install TensorFlow on 64-bit machines.
  * Added a short doc explaining how `Estimator`s save checkpoints.
  * Add documentation for ops supported by the `tf2xla` bridge.
  * Fix minor typos in the doc of `SpaceToDepth` and `DepthToSpace`.
  * Updated documentation comments in `mfcc_mel_filterbank.h` and `mfcc.h` to
    clarify that the input domain is squared magnitude spectra and the weighting
    is done on linear magnitude spectra (sqrt of inputs).
  * Change `tf.contrib.distributions` docstring examples to use `tfd` alias
    rather than `ds`, `bs`.
  * Fix docstring typos in `tf.distributions.bijectors.Bijector`.
  * `tf.assert_equal` no longer raises `ValueError.` It now raises
    `InvalidArgumentError,` as documented.
  * Update Getting Started docs and API intro.
* Google Cloud Storage (GCS):
  * Add userspace DNS caching for the GCS client.
  * Customize request timeouts for the GCS filesystem.
  * Improve GCS filesystem caching.
* Bug Fixes:
  * Fix bug where partitioned integer variables got their wrong shapes. Before
  * Fix correctness bug in CPU and GPU implementations of Adadelta.
  * Fix a bug in `import_meta_graph`'s handling of partitioned variables when
    importing into a scope. WARNING: This may break loading checkpoints of
    graphs with partitioned variables saved after using `import_meta_graph` with
    a non-empty `import_scope` argument.
  * Fix bug in offline debugger which prevented viewing events.
  * Added the `WorkerService.DeleteWorkerSession` method to the gRPC interface,
    to fix a memory leak. Ensure that your master and worker servers are running
    the same version of TensorFlow to avoid compatibility issues.
  * Fix bug in peephole implementation of BlockLSTM cell.
  * Fix bug by casting dtype of `log_det_jacobian` to match `log_prob` in
    `TransformedDistribution`.
  * Fix a bug in `import_meta_graph`'s handling of partitioned variables when
  * Ensure `tf.distributions.Multinomial` doesn't underflow in `log_prob`.
    Before this change, all partitions of an integer variable were initialized
    with the shape of the unpartitioned variable; after this change they are
    initialized correctly.
* Other:
  * Add necessary shape util support for bfloat16.
  * Add a way to run ops using a step function to MonitoredSession.
  * Add `DenseFlipout` probabilistic layer.
  * A new flag `ignore_live_threads` is available on train. If set to `True`, it
    will ignore threads that remain running when tearing down infrastructure
    after successfully completing training, instead of throwing a RuntimeError.
  * Restandardize `DenseVariational` as simpler template for other probabilistic
    layers.
  * `tf.data` now supports `tf.SparseTensor` components in dataset elements.
  * It is now possible to iterate over `Tensor`s.
  * Allow `SparseSegmentReduction` ops to have missing segment IDs.
  * Modify custom export strategy to account for multidimensional sparse float
    splits.
  * `Conv2D`, `Conv2DBackpropInput`, `Conv2DBackpropFilter` now supports arbitrary
    dilations with GPU and cuDNNv6 support.
  * `Estimator` now supports `Dataset`: `input_fn` can return a `Dataset`
    instead of `Tensor`s.
  * Add `RevBlock`, a memory-efficient implementation of reversible residual layers.
  * Reduce BFCAllocator internal fragmentation.
  * Add `cross_entropy` and `kl_divergence` to `tf.distributions.Distribution`.
  * Add `tf.nn.softmax_cross_entropy_with_logits_v2` which enables backprop
    w.r.t. the labels.
  * GPU back-end now uses `ptxas` to compile generated PTX.
  * `BufferAssignment`'s protocol buffer dump is now deterministic.
  * Change embedding op to use parallel version of `DynamicStitch`.
  * Add support for sparse multidimensional feature columns.
  * Speed up the case for sparse float columns that have only 1 value.
  * Allow sparse float splits to support multivalent feature columns.
  * Add `quantile` to `tf.distributions.TransformedDistribution`.
  * Add `NCHW_VECT_C` support for `tf.depth_to_space` on GPU.
  * Add `NCHW_VECT_C` support for `tf.space_to_depth` on GPU.

## API Changes
* Rename `SqueezeDims` attribute to `Axis` in C++ API for Squeeze op.
* `Stream::BlockHostUntilDone` now returns Status rather than bool.
* Minor refactor: move stats files from `stochastic` to `common` and remove
  `stochastic`.

## Known Bugs
* Using XLA:GPU with CUDA 9 and CUDA 9.1 results in garbage results and/or
  `CUDA_ILLEGAL_ADDRESS` failures.

  Google discovered in mid-December 2017 that the PTX-to-SASS compiler in CUDA 9
  and CUDA 9.1 sometimes does not properly compute the carry bit when
  decomposing 64-bit address calculations with large offsets (e.g. `load [x +
  large_constant]`) into 32-bit arithmetic in SASS.

  As a result, these versions of `ptxas` miscompile most XLA programs which use
  more than 4GB of temp memory.  This results in garbage results and/or
  `CUDA_ERROR_ILLEGAL_ADDRESS` failures.

  A fix in CUDA 9.1.121 is expected in late February 2018.  We do not expect a
  fix for CUDA 9.0.x.  Until the fix is available, the only workaround is to
  [downgrade](https://developer.nvidia.com/cuda-toolkit-archive) to CUDA 8.0.x
  or disable XLA:GPU.

  TensorFlow will print a warning if you use XLA:GPU with a known-bad version of
  CUDA; see e00ba24c4038e7644da417ddc639169b6ea59122.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Adam Zahran, Ag Ramesh, Alan Lee, Alan Yee, Alex Sergeev, Alexander, Amir H. Jadidinejad,
Amy, Anastasios Doumoulakis, Andrei Costinescu, Andrei Nigmatulin, Anthony Platanios,
Anush Elangovan, arixlin, Armen Donigian, ArtëM Sobolev, Atlas7, Ben Barsdell, Bill Prin,
Bo Wang, Brett Koonce, Cameron Thomas, Carl Thomé, Cem Eteke, cglewis, Changming Sun,
Charles Shenton, Chi-Hung, Chris Donahue, Chris Filo Gorgolewski, Chris Hoyean Song,
Chris Tava, Christian Grail, Christoph Boeddeker, cinqS, Clayne Robison, codrut3, concerttttt,
CQY, Dan Becker, Dan Jarvis, Daniel Zhang, David Norman, dmaclach, Dmitry Trifonov,
Donggeon Lim, dongpilYu, Dr. Kashif Rasul, Edd Wilder-James, Eric Lv, fcharras, Felix Abecassis,
FirefoxMetzger, formath, FredZhang, Gaojin Cao, Gary Deer, Guenther Schmuelling, Hanchen Li,
Hanmin Qin, hannesa2, hyunyoung2, Ilya Edrenkin, Jackson Kontny, Jan, Javier Luraschi,
Jay Young, Jayaram Bobba, Jeff, Jeff Carpenter, Jeremy Sharpe, Jeroen BéDorf, Jimmy Jia,
Jinze Bai, Jiongyan Zhang, Joe Castagneri, Johan Ju, Josh Varty, Julian Niedermeier,
JxKing, Karl Lessard, Kb Sriram, Keven Wang, Koan-Sin Tan, Kyle Mills, lanhin, LevineHuang,
Loki Der Quaeler, Loo Rong Jie, Luke Iwanski, LáSzló Csomor, Mahdi Abavisani, Mahmoud Abuzaina,
ManHyuk, Marek ŠUppa, MathSquared, Mats Linander, Matt Wytock, Matthew Daley, Maximilian Bachl,
mdymczyk, melvyniandrag, Michael Case, Mike Traynor, miqlas, Namrata-Ibm, Nathan Luehr,
Nathan Van Doorn, Noa Ezra, Nolan Liu, Oleg Zabluda, opensourcemattress, Ouwen Huang,
Paul Van Eck, peisong, Peng Yu, PinkySan, pks, powderluv, Qiao Hai-Jun, Qiao Longfei,
Rajendra Arora, Ralph Tang, resec, Robin Richtsfeld, Rohan Varma, Ryohei Kuroki, SaintNazaire,
Samuel He, Sandeep Dcunha, sandipmgiri, Sang Han, scott, Scott Mudge, Se-Won Kim, Simon Perkins,
Simone Cirillo, Steffen Schmitz, Suvojit Manna, Sylvus, Taehoon Lee, Ted Chang, Thomas Deegan,
Till Hoffmann, Tim, Toni Kunic, Toon Verstraelen, Tristan Rice, Urs KöSter, Utkarsh Upadhyay,
Vish (Ishaya) Abrams, Winnie Tsang, Yan Chen, Yan Facai (颜发才), Yi Yang, Yong Tang,
Youssef Hesham, Yuan (Terry) Tang, Zhengsheng Wei, zxcqwe4906, 张志豪, 田传武 

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 1.4.1

## Bug Fixes and Other Changes
* `LinearClassifier` fix.

# Release 1.4.0

## Major Features And Improvements
* `tf.keras` is now part of the core TensorFlow API.
* [`tf.data`](http://tensorflow.org/guide/datasets) is now part of
  the core TensorFlow API.
  * The API is now subject to backwards compatibility guarantees.
  * For a guide to migrating from the `tf.contrib.data` API, see the
    [README](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/data/README.md).
  * Major new features include `Dataset.from_generator()` (for building an input
    pipeline from a Python generator), and the `Dataset.apply()` method for
    applying custom transformation functions.
  * Several custom transformation functions have been added, including
    `tf.contrib.data.batch_and_drop_remainder()` and
    `tf.contrib.data.sloppy_interleave()`.
* Add `train_and_evaluate` for simple distributed `Estimator` training.
* Add `tf.spectral.dct` for computing the DCT-II.
* Add Mel-Frequency Cepstral Coefficient support to `tf.contrib.signal`
  (with GPU and gradient support).
* Add a self-check on `import tensorflow` for Windows DLL issues.
* Add NCHW support to `tf.depth_to_space` on GPU.
* TensorFlow Debugger (tfdbg):
  * Add `eval` command to allow evaluation of arbitrary Python/numpy expressions
    in tfdbg command-line interface. See
    [Debugging TensorFlow Programs](https://www.tensorflow.org/guide/debugger)
    for more details.
  * Usability improvement: The frequently used tensor filter `has_inf_or_nan` is
    now added to `Session` wrappers and hooks by default. So there is no need
    for clients to call `.add_tensor_filter(tf_debug.has_inf_or_nan)` anymore.
* SinhArcsinh (scalar) distribution added to `contrib.distributions`.
* Make `GANEstimator` opensource.
* `Estimator.export_savedmodel()` now includes all valid serving signatures
  that can be constructed from the Serving Input Receiver and all available
  ExportOutputs. For instance, a classifier may provide regression- and
  prediction-flavored outputs, in addition to the classification-flavored one.
  Building signatures from these allows TF Serving to honor requests using the
  different APIs (Classify, Regress, and Predict). Furthermore,
  `serving_input_receiver_fn()` may now specify alternative subsets of nodes
  that may act as inputs. This allows, for instance, producing a prediction
  signature for a classifier that accepts raw `Tensors` instead of a serialized
  `tf.Example`.
* Add `tf.contrib.bayesflow.hmc`.
* Add `tf.contrib.distributions.MixtureSameFamily`.
* Make `Dataset.shuffle()` always reshuffles after each iteration by default.
* Add `tf.contrib.bayesflow.metropolis_hastings`.
* Add `log_rate` parameter to `tf.contrib.distributions.Poisson`.
* Extend `tf.contrib.distributions.bijector` API to handle some non-injective
  transforms.
* Java:
  * Generics (e.g., `Tensor<Integer>`) for improved type-safety
    (courtesy @andrewcmyers).
  * Support for multi-dimensional string tensors.
  * Support loading of custom operations (e.g. many in `tf.contrib`) on Linux
    and OS X
* All our prebuilt binaries have been built with CUDA 8 and cuDNN 6.
  We anticipate releasing TensorFlow 1.5 with CUDA 9 and cuDNN 7.

## Bug Fixes and Other Changes
* `tf.nn.rnn_cell.DropoutWrapper` is now more careful about dropping out LSTM
  states.  Specifically, it no longer ever drops the `c` (memory) state of an
  `LSTMStateTuple`.  The new behavior leads to proper dropout behavior
  for LSTMs and stacked LSTMs.  This bug fix follows recommendations from
  published literature, but is a behavioral change.  State dropout behavior
  may be customized via the new `dropout_state_filter_visitor` argument.
* Removed `tf.contrib.training.python_input`.  The same behavior, in a more
  flexible and reproducible package, is available via the new
  `tf.contrib.data.Dataset.from_generator` method!
* Fix `tf.contrib.distributions.Affine` incorrectly computing log-det-jacobian.
* Fix `tf.random_gamma` incorrectly handling non-batch, scalar draws.
* Resolved a race condition in TensorForest TreePredictionsV4Op.
* Google Cloud Storage file system, Amazon S3 file system, and Hadoop file
  system support are now default build options.
* Custom op libraries must link against libtensorflow_framework.so
  (installed at `tf.sysconfig.get_lib()`).
* Change `RunConfig` default behavior to not set a random seed, making random
  behavior independently random on distributed workers. We expect this to
  generally improve training performance. Models that do rely on determinism
  should set a random seed explicitly.

## Breaking Changes to the API
* The signature of the `tf.contrib.data.rejection_resample()` function has been
  changed. It now returns a function that can be used as an argument to
  `Dataset.apply()`.
* Remove `tf.contrib.data.Iterator.from_dataset()` method. Use
  `Dataset.make_initializable_iterator()` instead.
* Remove seldom used and unnecessary `tf.contrib.data.Iterator.dispose_op()`.
* Reorder some TFGAN loss functions in a non-backwards compatible way.

## Known Issues
* In Python 3, `Dataset.from_generator()` does not support Unicode strings.
  You must convert any strings to bytes objects before yielding them from
  the generator.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

4d55397500, Abdullah Alrasheed, abenmao, Adam Salvail, Aditya Dhulipala, Ag Ramesh,
Akimasa Kimura, Alan Du, Alan Yee, Alexander, Amit Kushwaha, Amy, Andrei Costinescu,
Andrei Nigmatulin, Andrew Erlichson, Andrew Myers, Andrew Stepanov, Androbin, AngryPowman,
Anish Shah, Anton Daitche, Artsiom Chapialiou, asdf2014, Aseem Raj Baranwal, Ash Hall,
Bart Kiers, Batchu Venkat Vishal, ben, Ben Barsdell, Bill Piel, Carl Thomé, Catalin Voss,
Changming Sun, Chengzhi Chen, Chi Zeng, Chris Antaki, Chris Donahue, Chris Oelmueller,
Chris Tava, Clayne Robison, Codrut, Courtial Florian, Dalmo Cirne, Dan J, Darren Garvey,
David Kristoffersson, David Norman, David RöThlisberger, DavidNorman, Dhruv, DimanNe,
Dorokhov, Duncan Mac-Vicar P, EdwardDixon, EMCP, error.d, FAIJUL, Fan Xia,
Francois Xavier, Fred Reiss, Freedom" Koan-Sin Tan, Fritz Obermeyer, Gao, Xiang,
Guenther Schmuelling, Guo Yejun (郭叶军), Hans Gaiser, HectorSVC, Hyungsuk Yoon,
James Pruegsanusak, Jay Young, Jean Wanka, Jeff Carpenter, Jeremy Rutman, Jeroen BéDorf,
Jett Jones, Jimmy Jia, jinghuangintel, jinze1994, JKurland, Joel Hestness, joetoth,
John B Nelson, John Impallomeni, John Lawson, Jonas, Jonathan Dekhtiar, joshkyh, Jun Luan,
Jun Mei, Kai Sasaki, Karl Lessard, karl@kubx.ca, Kb Sriram, Kenichi Ueno, Kevin Slagle,
Kongsea, Lakshay Garg, lhlmgr, Lin Min, liu.guangcong, Loki Der Quaeler, Louie Helm,
lucasmoura, Luke Iwanski, Lyndon White, Mahmoud Abuzaina, Marcel Puyat, Mark Aaron Shirley,
Michele Colombo, MtDersvan, Namrata-Ibm, Nathan Luehr, Naurril, Nayana Thorat, Nicolas Lopez,
Niranjan Hasabnis, Nolan Liu, Nouce, Oliver Hennigh, osdamv, Patrik Erdes,
Patryk Chrabaszcz, Pavel Christof, Penghao Cen, postBG, Qingqing Cao, Qingying Chen, qjivy,
Raphael, Rasmi, raymondxyang, Renze Yu, resec, Roffel, Ruben Vereecken, Ryohei Kuroki,
sandipmgiri, Santiago Castro, Scott Kirkland, Sean Vig, Sebastian Raschka, Sebastian Weiss,
Sergey Kolesnikov, Sergii Khomenko, Shahid, Shivam Kotwalia, Stuart Berg, Sumit Gouthaman,
superzerg, Sven Mayer, tetris, Ti Zhou, Tiago Freitas Pereira, Tian Jin, Tomoaki Oiki,
Vaibhav Sood, vfdev, Vivek Rane, Vladimir Moskva, wangqr, Weber Xie, Will Frey,
Yan Facai (颜发才), yanivbl6, Yaroslav Bulatov, Yixing Lao, Yong Tang, youkaichao,
Yuan (Terry) Tang, Yue Zhang, Yuxin Wu, Ziming Dong, ZxYuan, 黄璞

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 1.3.0

See also [TensorBoard 0.1.4](https://github.com/tensorflow/tensorboard/releases/tag/0.1.4) release notes.

## Major Features and Improvements
* Added canned estimators to Tensorflow library. List of added estimators:
  * `DNNClassifier`
  * `DNNRegressor`
  * `LinearClassifier`
  * `LinearRegressor`
  * `DNNLinearCombinedClassifier`
  * `DNNLinearCombinedRegressor`.
* All our prebuilt binaries have been built with cuDNN 6. We anticipate releasing TensorFlow 1.4 with cuDNN 7.
* `import tensorflow` now goes much faster.
* Adds a file cache to the GCS filesystem with configurable max staleness for file contents. This permits caching of file contents across close/open boundaries.
* Added an axis parameter to `tf.gather`.
* Added a `constant_values` keyword argument to `tf.pad`.
* Adds `Dataset.interleave` transformation.
* Add `ConcatenateDataset` to concatenate two datasets.
* Added Mobilenet support to TensorFlow for Poets training script.
* Adds a block cache to the GCS filesystem with configurable block size and count.
* SinhArcSinh bijector added.
* Added `Dataset.list_files` API.
* Introduces new operations and Python bindings for the Cloud TPU.
* Adding TensorFlow-iOS CocoaPod for symmetry with tensorflow-android.
* Introduces base implementations of ClusterResolvers.
* Unify memory representations of TensorShape and PartialTensorShape. As a consequence, tensors now have a maximum of 254 dimensions, not 255.
* Changed references to LIBXSMM to use version 1.8.1.
* TensorFlow Debugger (tfdbg):
  * Display summaries of numeric tensor values with the `-s` flag to command `print_tensor` or `pt`.
  * Display feed values with the `print_feed` or `pf` command and clickable links in the curses UI.
  * Runtime profiler at the op level and the Python source line level with the `run -p` command.
* Initial release of the statistical distribution library `tf.distributions`.
* GPU kernels and speed improvements for unary `tf.where` and `tf.nn.top_k`.
* Monotonic Attention wrappers added to `tf.contrib.seq2seq`.
* Added `tf.contrib.signal`, a library for signal processing primitives.
* Added `tf.contrib.resampler`, containing CPU and GPU ops for differentiable resampling of images.

## Breaking Changes to the API
* `tf.RewriterConfig` was removed from the Python API after being available in 1.2 release candidates (it was never in an actual release). Graph rewriting is still available, just not as `tf.RewriterConfig`. Instead add an explicit import.
* Breaking change to `tf.contrib.data.Dataset` APIs that expect a nested structure. Lists are now converted to `tf.Tensor` implicitly. You may need to change uses of lists to tuples in existing code. In addition, dicts are now supported as a nested structure.

## Changes to contrib APIs
* Adds tf.contrib.nn.rank_sampled_softmax_loss, a sampled-softmax variant that can improve rank loss.
* `tf.contrib.metrics`.{streaming_covariance,streaming_pearson_correlation} modified to return nan when they have seen less or equal to 1 unit of weight.
* Adds time series models to contrib. See contrib/timeseries/README.md for details.
* Adds FULLY_CONNECTED Op to tensorflow/contrib/lite/schema.fbs

## Known Issues
* Tensorflow_gpu compilation fails with Bazel 0.5.3.

## Bug Fixes and Other Changes
* Fixes `strides` and `begin` dtype mismatch when slicing using int64 Tensor index in python.
* Improved convolution padding documentation.
* Add a tag constant, gpu, to present graph with GPU support.
* `saved_model.utils` now support SparseTensors transparently.
* A more efficient implementation of non-max suppression.
* Add support for the shrinkage-type L2 to FtrlOptimizer in addition to the online L2 it already supports.
* Fix negative variance in moments calculation.
* Expand UniqueOp Benchmark Tests to cover more collision cases.
* Improves stability of GCS filesystem on Mac.
* Add time estimation to HloCostAnalysis.
* Fixed the bug in Estimator that params in constructor was not a deepcopy of the user provided one. This bugs inadvertently enabled user to mutate the params after the creation of Estimator, leading to potentially undefined behavior.
* Added None check for save_path in `saver.restore`.
* Register devices under their legacy names in device_mgr to ease the transition to clusterspec-propagated configurations.
* VectorExponential added to distributions.
* Add a bitwise module with bitwise_and, bitwise_or, bitwise_xor, and invert functions.
* Add fixed-grid ODE integration routines.
* Allow passing bounds to ScipyOptimizerInterface.
* Correctness fixes for fft_length parameter to `tf.spectral.rfft` & `tf.spectral.irfft`.
* Exported model signatures using the 'predict' method will no longer have their input and output keys silently ignored and rewritten to 'inputs' and 'outputs'. If a model was exported with different names before 1.2, and is now served with tensorflow/serving, it will accept requests using 'inputs' and 'outputs'. Starting at 1.2, such a model will accept the keys specified during export. Therefore, inference requests using 'inputs' and 'outputs' may start to fail. To fix this, either update any inference clients to send requests with the actual input and output keys used by the trainer code, or conversely, update the trainer code to name the input and output Tensors 'inputs' and 'outputs', respectively. Signatures using the 'classify' and 'regress' methods are not affected by this change; they will continue to standardize their input and output keys as before.
* Add in-memory caching to the Dataset API.
* Set default end_of_sequence variable in datasets iterators to false.
* [Performance] Increase performance of `tf.layers.conv2d` when setting use_bias=True by 2x by using nn.bias_add.
* Update iOS examples to use CocoaPods, and moved to tensorflow/examples/ios.
* Adds a family= attribute in `tf.summary` ops to allow controlling the tab name used in Tensorboard for organizing summaries.
* When GPU is configured, do not require --config=cuda, instead, automatically build for GPU if this is requested in the configure script.
* Fix incorrect sampling of small probabilities in CPU/GPU multinomial.
* Add a list_devices() API on sessions to list devices within a cluster. Additionally, this change augment the ListDevices master API to support specifying a session.
* Allow uses of over-parameterized separable convolution.
* TensorForest multi-regression bug fix.
* Framework now supports armv7, cocoapods.org now displays correct page.
* Script to create iOS framework for CocoaPods.
* Android releases of TensorFlow are now pushed to jcenter for easier integration into apps. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/android/README.md for more details.
* TensorFlow Debugger (tfdbg):
  * Fixed a bug that prevented tfdbg from functioning with multi-GPU setups.
  * Fixed a bug that prevented tfdbg from working with `tf.Session.make_callable`.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

4F2E4A2E, Adriano Carmezim, Adrià Arrufat, Alan Yee, Alex Lattas, Alex Rothberg,
Alexandr Baranezky, Ali Siddiqui, Andreas Solleder, Andrei Costinescu, Andrew Hundt,
Androbin, Andy Kernahan, Anish Shah, Anthony Platanios, Arvinds-Ds, b1rd, Baptiste
Arnaud, Ben Mabey, Benedikt Linse, Beomsu Kim, Bo Wang, Boyuan Deng, Brett Koonce,
Bruno Rosa, Carl Thomé, Changming Sun, Chase Roberts, Chirag Bhatia, Chris Antaki,
Chris Hoyean Song, Chris Tava, Christos Nikolaou, Croath Liu, cxx, Czxck001, Daniel
Ylitalo, Danny Goodman, Darren Garvey, David Brailovsky, David Norman, DavidNorman,
davidpham87, ddurham2, Dhruv, DimanNe, Drew Hintz, Dustin Tran, Earthson Lu, ethiraj,
Fabian Winnen, Fei Sun, Freedom" Koan-Sin Tan, Fritz Obermeyer, Gao, Xiang, Gautam,
Guenther Schmuelling, Gyu-Ho Lee, Hauke Brammer, horance, Humanity123, J Alammar,
Jayeol Chun, Jeroen BéDorf, Jianfei Wang, jiefangxuanyan, Jing Jun Yin, Joan Puigcerver,
Joel Hestness, Johannes Mayer, John Lawson, Johnson145, Jon Malmaud, Jonathan Alvarez-Gutierrez,
Juang, Yi-Lin, Julian Viereck, Kaarthik Sivashanmugam, Karl Lessard, karl@kubx.ca, Kevin
Carbone, Kevin Van Der Burgt, Kongsea, ksellesk, lanhin, Lef Ioannidis, Liangliang He,
Louis Tiao, Luke Iwanski, LáSzló Csomor, magixsno, Mahmoud Abuzaina, Marcel Hlopko, Mark
Neumann, Maxwell Paul Brickner, mdfaijul, MichaëL Defferrard, Michał JastrzęBski, Michele
Colombo, Mike Brodie, Mosnoi Ion, mouradmourafiq, myPrecious, Nayana Thorat,
Neeraj Kashyap, Nelson Liu, Niranjan Hasabnis, Olivier Moindrot, orome, Pankaj Gupta, Paul
Van Eck, peeyush18, Peng Yu, Pierre, preciousdp11, qjivy, Raingo, raoqiyu, ribx, Richard S.
Imaoka, Rishabh Patel, Robert Walecki, Rockford Wei, Ryan Kung, Sahil Dua, Sandip Giri, Sayed
Hadi Hashemi, sgt101, Shitian Ni, Shuolongbj, Siim PõDer, Simon Perkins, sj6077, SOLARIS,
Spotlight0xff, Steffen Eberbach, Stephen Fox, superryanguo, Sven Mayer, Tapan Prakash,
Tiago Morais Morgado, Till Hoffmann, Tj Rana, Vadim Markovtsev, vhasanov, Wei Wu,
windead, Yan (Asta) Li, Yan Chen, Yann Henon, Yi Wang, Yong Tang, yorkie, Yuan (Terry)
Tang, Yuxin Wu, zhengjiajin, zhongzyd, 黄璞

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 1.2.1

## Bug Fixes and Other Changes
* Updating markdown version required to >= 2.6.8.
* Support tensors as dropout rates again, by removing the min(max(..))

# Release 1.2.0

## Major Features and Improvements
* Python 3.6 support on Windows.
* Added `tf.layers.conv3d_transpose` layer for spatio temporal deconvolution.
* Added `tf.Session.make_callable()`, which provides a lower overhead means of running a similar step multiple times.
* Added libverbs-based RDMA support to contrib (courtesy @junshi15 from Yahoo).
* Bring `tf.feature_column.*` into the API. Non-deprecated functionality from `tf.contrib.layers.*` is moved to `tf.feature_column.*` with cosmetic changes.
* `RNNCell` objects now subclass `tf.layers.Layer`.  The strictness described
  in the TensorFlow 1.1 release is gone:  The first time an RNNCell is used,
  it caches its scope.  All future uses of the RNNCell will reuse variables from
  that same scope.  This is a breaking change from the behavior of RNNCells
  in TensorFlow versions <= 1.0.1.  TensorFlow 1.1 had checks in place to
  ensure old code works correctly with the new semantics; this version
  allows more flexible uses of RNNCell but can lead to subtle errors if
  using code meant for TensorFlow <= 1.0.1.  For example, writing:
  `MultiRNNCell([lstm] * 5)` will now build a 5-layer LSTM stack where each
  layer shares the **same** parameters.  To get 5 layers each with their own
  parameters, write: `MultiRNNCell([LSTMCell(...) for _ in range(5)])`.
  If at all unsure, first test your code with TF 1.1; ensure it raises no
  errors, and then upgrade to TF 1.2.
* RNNCells' variable names have been renamed for consistency with Keras layers.
  Specifically, the previous variable names "weights" and "biases" have
  been changed to "kernel" and "bias", respectively.
  This may cause backward incompatibility with regard to your old
  checkpoints containing such RNN cells, in which case you can use the tool
  [checkpoint_convert script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/tools/checkpoint_convert.py)
  to convert the variable names in your old checkpoints.
* Many of the RNN functions and classes that were in the `tf.nn` namespace
  before the 1.0 release and which were moved to `tf.contrib.rnn` have now
  been moved back to the core namespace.  This includes
  `RNNCell`, `LSTMCell`, `GRUCell`, and a number of other cells.  These
  now reside in `tf.nn.rnn_cell` (with aliases in `tf.contrib.rnn` for backwards
  compatibility).  The original `tf.nn.rnn` function is now `tf.nn.static_rnn`,
  and the bidirectional static and state saving static rnn functions are also
  now back in the `tf.nn` namespace.

  Notable exceptions are the `EmbeddingWrapper`, `InputProjectionWrapper` and
  `OutputProjectionWrapper`,  which will slowly be moved to deprecation
  in `tf.contrib.rnn`.  These are inefficient wrappers that should often
  be replaced by calling `embedding_lookup` or `layers.dense` as pre- or post-
  processing of the rnn.  For RNN decoding, this functionality has been replaced
  with an alternative API in `tf.contrib.seq2seq`.
* Intel MKL Integration (https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture). Intel developed a number of
  optimized deep learning primitives: In addition to matrix multiplication and
  convolution, these building blocks include:
  Direct batched convolution
  Pooling: maximum, minimum, average
  Normalization: LRN, batch normalization
  Activation: rectified linear unit (ReLU)
  Data manipulation: multi-dimensional transposition (conversion), split,
  concat, sum and scale.
* TensorForest Estimator now supports SavedModel export for serving.
* Support client-provided ClusterSpec's and propagate them to all workers to enable the creation of dynamic TensorFlow clusters.
* TensorFlow C library now available for Windows.
* We released a new open-source version of TensorBoard.
* [`SavedModel CLI`](https://www.tensorflow.org/versions/master/guide/saved_model_cli) tool available to inspect and execute MetaGraph in SavedModel
* Android releases of TensorFlow are now pushed to jcenter for easier
  integration into apps. See
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/android/README.md
  for more details.

## Deprecations

* TensorFlow 1.2 may be the last time we build with cuDNN 5.1. Starting with
  TensorFlow 1.3, we will try to build all our prebuilt binaries with cuDNN 6.0.
  While we will try to keep our source code compatible with cuDNN 5.1, it will
  be best effort.

## Breaking Changes to the API
* `org.tensorflow.contrib.android.TensorFlowInferenceInterface` now throws exceptions where possible and has simplified method signatures.

## Changes to contrib APIs
* Added `tf.contrib.util.create_example`.
* Added bilinear interpolation to `tf.contrib.image`.
* Add `tf.contrib.stateless` for random ops with custom seed control.
* MultivariateNormalFullCovariance added to contrib/distributions/
* tensorflow/contrib/rnn undergoes RNN cell variable renaming for
  consistency with Keras layers. Specifically, the previous variable names
  "weights" and "biases" are changed to "kernel" and "bias", respectively.
  This may cause backward incompatibility with regard to your old
  checkpoints containing such RNN cells, in which case you can use the
  [checkpoint_convert script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/tools/checkpoint_convert.py)
  to convert the variable names in your old checkpoints.
* Added `tf.contrib.kernel_methods` module with Ops and estimators for primal
  (explicit) kernel methods in TensorFlow.

## Bug Fixes and Other Changes
* In python, `Operation.get_attr` on type attributes returns the Python DType
  version of the type to match expected get_attr documentation rather than the
  protobuf enum.
* tensorflow/contrib/rnn undergoes RNN cell variable renaming for
  consistency with Keras layers. Specifically, the previous variable names
  "weights" and "biases" are changed to "kernel" and "bias", respectively.
* Changed MIN_SDK version to 8.0 when building iOS libraries.
* Fixed LIBXSMM integration.
* Make decode_jpeg/decode_png/decode_gif handle all formats, since users frequently try to decode an image as the wrong type.
* Improve implicit broadcasting lowering.
* Improving stability of GCS/BigQuery clients by a faster retrying of stale transmissions.
* Remove OpKernelConstruction::op_def() as part of minimizing proto dependencies.
* VectorLaplaceDiag distribution added.
* Android demo no longer requires libtensorflow_demo.so to run (libtensorflow_inference.so still required)
* Added `categorical_column_with_vocabulary_file`.
* Introduce ops for batching/unbatching tensors across Session::Run() calls.
* Add tf.log_sigmoid(x) = tf.log(tf.sigmoid(x)) = -tf.nn.softplus(-x).
* Changed hooks lists to immutable tuples, and now allow any iterable for the associated arguments.
* Introduce TFDecorator.
* Added an Mfcc op for speech feature generation.
* Improved DirectSession::Run() overhead and error checking. Feeding a value of the wrong type will now synchronously raise an INVALID_ARGUMENT error instead of asynchronously raising an INTERNAL error. Code that depends on the (undefined) behavior when feeding a tensor of the wrong type may need to be updated.
* Added unreduced NONE, and reduced MEAN options for losses. Removed "WEIGHTED_" prefix from other Reduction constants.
* assertAllClose now handles dicts.
* Added Gmock matcher for HloInstructions.
* Add var name to errors on variable restore.
* Added an AudioSpectrogram op for audio feature generation.
* Added `reduction` arg to losses.
* `tf.placeholder` can represent scalar shapes and partially known.
* Remove estimator_spec(mode) argument.
* Added an AudioSpectrogram op for audio feature generation.
* TensorBoard disables all runs by default if there are more than 40 runs.
* Removed old doc generator code.
* GCS file system integration now supports domain buckets, e.g gs://bucket.domain.com/path.
* Add `tf.summary.text` for outputting text to TensorBoard.
* The "run" command of tfdbg's command-line interface now supports filtering of tensors by node name, op type and tensor dtype.
* `tf.string_to_number` now supports int64 and float64 outputs.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

4F2E4A2E, Aaron Schumacher, Abhi Agg, admcrae, Adriano Carmezim, Adrià Arrufat,
agramesh1, Akimitsu Seo, Alan Mosca, Alex Egg, Alex Rothberg, Alexander Heinecke,
Alexander Matyasko, Alexandr Baranezky, Alexandre Caulier, Ali Siddiqui, Anand Venkat,
Andrew Hundt, Androbin, Anmol Sharma, Arie, Arno Leist, Arron Cao, AuréLien Geron, Bairen Yi,
Beomsu Kim, Carl Thomé, cfperez, Changming Sun, Corey Wharton, critiqjo, Dalei Li, Daniel
Rasmussen, Daniel Trebbien, DaríO Hereñú, David Eng, David Norman, David Y. Zhang, Davy Song, ddurham2,
Deepak Subburam, Dmytro Kyrychuk, Dominic Rossi, Dominik SchlöSser, Dustin Tran,
Eduardo Pinho, Egil Martinsson, Elliot Saba, Eric Bigelow, Erik Smistad, Evan Klitzke,
Fabrizio Milo, Falcon Dai, Fei Gao, FloopCZ, Fung Lam, Gautam, GBLin5566, Greg Peatfield,
Gu Wang, Guenther Schmuelling, Hans Pabst, Harun Gunaydin, Huaizheng, Ido Shamay, Ikaro
Silva, Ilya Edrenkin, Immexxx, James Mishra, Jamie Cooke, Jay Young, Jayaram Bobba,
Jianfei Wang, jinghua2, Joey Meyer, John Maidens, Jonghoon Jin, Julian Villella,
Jun Kim, Jun Shi, Junwei Pan, jyegerlehner, Karan Desai, Karel Van De Plassche,
Kb Sriram, KhabarlakKonstantin, Koan-Sin Tan, krivard, Kwotsin, Leandro Gracia Gil,
Li Chen, Liangliang He, Louie Helm, lspvic, Luiz Henrique Soares, LáSzló Csomor,
Mark Wong, Mathew Wicks, Matthew Rahtz, Maxwell Paul Brickner, Michael Hofmann, Miguel
Flores Ruiz De Eguino, MikeTam1021, Mortada Mehyar, Mycosynth, Namnamseo,
Nate Harada, Neven Miculinic, Nghia Tran, Nick Lyu, Niranjan Hasabnis, Nishidha, Oleksii
Kuchaiev, Oyesh Mann Singh, Panmari, Patrick, Paul Van Eck, Piyush Chaudhary, Quim Llimona,
Raingo, Richard Davies, Ruben Vereecken, Sahit Chintalapudi, Sam Abrahams, Santiago Castro,
Scott Sievert, Sean O'Keefe, Sebastian Schlecht, Shane, Shubhankar Deshpande, Spencer Schaber,
Sunyeop Lee, t13m, td2014, Thomas H. P. Andersen, Toby Petty, Umang Mehta,
Vadim Markovtsev, Valentin Iovene, Vincent Zhao, Vit Stepanovs, Vivek Rane, Vu Pham, wannabesrevenge,
weipingpku, wuhaixutab, wydwww, Xiang Gao, Xiaolin Lin, xiaoyaozhuzi, Yaroslav Bulatov, Yi Liu,
Yoshihiro Sugi, Yuan (Terry) Tang, Yuming Wang, Yuxin Wu, Zader Zheng, Zhaojun Zhang, zhengjiajin,
ZhipengShen, Ziming Dong, zjj2wry

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 1.1.0

## Major Features and Improvements
* Added Java API support for Windows.
* Added `tf.spectral` module. Moved existing FFT ops to `tf.spectral` while
  keeping an alias in the old location (`tf.*`).
* Added 1D, 2D and 3D Fourier transform ops for real signals to `tf.spectral`.
* Added a `tf.bincount` function.
* Added Keras 2 API to contrib.
* Added a new lightweight queue-like object - `RecordInput`.
* Added `tf.contrib.image.compose_transforms` function.
* Bring `tf.estimator.*` into the API. Non-deprecated functionality from `tf.contrib.learn.Estimator` is moved to `tf.estimator.Estimator` with cosmetic changes.
* Docker images: TF images on gcr.io and Docker Hub are upgraded to ubuntu:16.04.
* Added the following features to TensorFlow Debugger (tfdbg):
  * Ability to inspect Python source file against TF ops and tensors (command `print_source` / `ps`)
  * New navigation bar in Curses-based UI
  * NodeStepper (command `invoke_stepper`) now uses intermediate tensor dumps. It also uses `TensorHandles` as direct feeds during successive `cont` calls for improved performance and reduced memory consumption.
* Initial release of installation guides for Java, C, and Go.
* Added Text Dashboard to TensorBoard.

## Deprecations

* TensorFlow 1.1.0 will be the last time we release a binary with Mac GPU support. Going forward, we will stop testing on Mac GPU systems. We continue to welcome patches that maintain Mac GPU support, and we will try to keep the Mac GPU build working.

## Changes to contrib APIs
* The behavior of RNNCells is now stricter due to the transition towards making RNNCells act more like Keras layers.
  * If an RNNCell is used twice in two different variable scopes, an error is raised describing how to avoid this behavior.
  * If an RNNCell is used in a variable scope with existing conflicting variables, an error is raised showing that the RNNCell must be constructed with argument `reuse=True`.
* Deprecated contrib/distributions `pmf`, `pdf`, `log_pmf`, `log_pdf`.
* Moved `bayesflow.special_math` to distributions.
* `tf.contrib.tensor_forest.python.tensor_forest.RandomForestDeviceAssigner` removed.
* Changed some MVN classes and parameters:
  * `tf.contrib.distributions.MultivariateNormalFull` replaced by `tf.contrib.distributions.MultivariateNormalTriL`.
  * `tf.contrib.distributions.MultivariateNormalCholesky` replaced by `tf.contrib.distributions.MultivariateNormalTriL`
  * `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev` replaced
    by `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusScale`
  * `tf.contrib.distributions.MultivariateNormalDiag` arguments changed from `mu`, `diag_stddev` to `log`, `scale_diag`.
  * `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT` removed.
  * `tf.contrib.distributions.MultivariateNormalDiagPlusLowRank` added.

## Bug Fixes and Other Changes
* Java: Support for loading models exported using the SavedModel API (courtesy @EronWright).
* Go: Added support for incremental graph execution.
* Fix a bug in the WALS solver when single-threaded.
* Added support for integer sparse feature values in `tf.contrib.layers.sparse_column_with_keys`.
* Fixed `tf.set_random_seed(0)` to be deterministic for all ops.
* Stability improvements for the GCS file system support.
* Improved TensorForest performance.
* Added support for multiple filename globs in `tf.matching_files`.
* `LogMessage` now includes a timestamp as beginning of a message.
* Added MultiBox person detector example standalone binary.
* Android demo: Makefile build functionality added to build.gradle to fully support building TensorFlow demo in Android on Windows.
* Android demo: read MultiBox priors from txt file rather than protobuf.
* Added colocation constraints to `StagingArea`.
* `sparse_matmul_op` reenabled for Android builds.
* Restrict weights rank to be the same as the broadcast target, to avoid ambiguity on broadcast rules.
* Upgraded libxsmm to 1.7.1 and applied other changes for performance and memory usage.
* Fixed bfloat16 integration of LIBXSMM sparse mat-mul.
* Improved performance and reduce memory usage by allowing ops to forward input buffers to output buffers and perform computations in-place.
* Improved the performance of CPU assignment for strings.
* Speed up matrix * vector multiplication and matrix * matrix with unknown shapes.
* C API: Graph imports now support input remapping, control dependencies, and returning imported nodes (see `TF_GraphImportGraphDefWithReturnOutputs()`)
* Multiple C++ API updates.
* Multiple TensorBoard updates including:
  * Users can now view image summaries at various sampled steps (instead of just the last step).
  * Bugs involving switching runs as well as the image dashboard are fixed.
  * Removed data download links from TensorBoard.
  * TensorBoard uses a relative data directory, for easier embedding.
  * TensorBoard automatically ignores outliers for domain calculation, and formats proportional values consistently.
* Multiple tfdbg bug fixes:
  * Fixed Windows compatibility issues.
  * Command history now persists across runs.
  * Bug fix in graph validation related to `tf.while_loops`.
* Java Maven fixes for bugs with Windows installation.
* Backport fixes and improvements from external keras.
* Keras config file handling fix.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

A. Besir Kurtulmus, Adal Chiriliuc, @akash, Alec-Desouza, Alex Rothberg, Alex
Sergeev, Alexander Heinecke, Allen Guo, Andreas Madsen, Ankesh Anand, Anton
Loss, @Aravind, @Arie, Ashutosh Das, AuréLien Geron, Bairen Yi, @bakunyo, Ben
Visser, Brady Zhou, Calpa Liu, Changming Sun, Chih Cheng Liang, Christopher
Berner, Clark Zinzow, @Conchylicultor, Dan Ellis, Dan J, Dan Jarvis, Daniel
Ylitalo, Darren Garvey, David Norman, David Truong, @DavidNorman, Dimitar
Pavlov, Dmitry Persiyanov, @Eddie, @elirex, Erfan Noury, Eron Wright, Evgeny
Mazovetskiy, Fabrizio (Misto) Milo, @fanlu, Fisher Coder, Florian Courtial,
Franck Dernoncourt, Gagan Goel, Gao, Xiang, @Gautam, Gefu Tang, @guilherme,
@guschmue, Hannah Provenza, Hans Pabst, @hartb, Hsiao Yi, Huazuo Gao, Igor
ChorążEwicz, Ivan Smirnov, Jakub Kolodziejczyk, Jason Gavris, Jason Morton, Jay
Young, Jayaram Bobba, Jeremy Sawruk, Jiaming Liu, Jihun Choi, @jiqiu, Joan Thibault,
John C F, Jojy George Varghese, Jon Malmaud, Julian Berman, Julian Niedermeier,
Junpeng Lao, Kai Sasaki, @Kankroc, Karl Lessard, Kyle Bostelmann, @Lezcano, Li
Yi, Luo Yun, @lurker, Mahmoud-Abuzaina, Mandeep Singh, Marek Kolodziej, Mark
Szepieniec, Martial Hue, Medhat Omr, Memo Akten, Michael Gharbi, MichaëL Defferrard,
Milan Straka, @MircoT, @mlucool, Muammar Ibn Faisal, Nayana Thorat, @nghiattran,
Nicholas Connor, Nikolaas Steenbergen, Niraj Patel, Niranjan Hasabnis, @Panmari,
Pavel Bulanov, Philip Pries Henningsen, Philipp Jund, @polonez, Prayag Verma, Rahul
Kavi, Raphael Gontijo Lopes, @rasbt, Raven Iqqe, Reid Pryzant, Richard Shin, Rizwan
Asif, Russell Kaplan, Ryo Asakura, RüDiger Busche, Saisai Shao, Sam Abrahams, @sanosay,
Sean Papay, @seaotterman, @selay01, Shaurya Sharma, Sriram Narayanamoorthy, Stefano
Probst, @taknevski, @tbonza, @teldridge11, Tim Anglade, Tomas Reimers, Tomer Gafner,
Valentin Iovene, Vamsi Sripathi, Viktor Malyi, Vit Stepanovs, Vivek Rane, Vlad Firoiu,
@wangg12, @will, Xiaoyu Tao, Yaroslav Bulatov, Yi Liu, Yuan (Terry) Tang, @Yufeng,
Yuming Wang, Yuxin Wu, Zafar Takhirov, Ziming Dong

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


# Release 1.0.1

## Bug Fixes and Other Changes
* Change GraphConstructor to not increase the version when importing, but instead take the min of all versions.
* Google Cloud Storage fixes.
* Removed `tf.core` and `tf.python` modules from the API. These were never intended to be exposed. Please use the same objects through top-level `tf` module instead.

# Release 1.0.0

## Major Features and Improvements
* XLA (experimental): initial release of [XLA](https://www.tensorflow.org/versions/master/experimental/xla/), a domain-specific compiler for TensorFlow graphs, that targets CPUs and GPUs.
* TensorFlow Debugger (tfdbg): command-line interface and API.
* New python 3 docker images added.
* Made pip packages pypi compliant. TensorFlow can now be installed by `pip
  install tensorflow` command.
* Several python API calls have been changed to resemble NumPy more closely.
* Android: person detection + tracking demo implementing Scalable Object
  Detection using Deep Neural Networks.
* New (experimental) [Java API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java).
* Add new Android image stylization demo based on "A Learned Representation For Artistic Style", and add YOLO object detector support.

## Breaking Changes to the API
To help you upgrade your existing TensorFlow Python code to match the API changes below, we have prepared a [conversion script](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility).
* TensorFlow/models have been moved to a separate github repository.
* Division and modulus operators (/, //, %) now match Python (flooring)
  semantics. This applies to `tf.div` and `tf.mod` as well. To obtain forced
  integer truncation based behaviors you can use `tf.truncatediv`
  and `tf.truncatemod`.
* `tf.divide()` is now the recommended division function. `tf.div()` will
  remain, but its semantics do not respond to Python 3 or `from future`
  mechanisms.
* tf.reverse() now takes indices of axes to be reversed. E.g.
  `tf.reverse(a, [True, False, True])` must now be written as
  `tf.reverse(a, [0, 2])`. `tf.reverse_v2()` will remain until 1.0 final.
* `tf.mul`, `tf.sub` and `tf.neg` are deprecated in favor of `tf.multiply`,
  `tf.subtract` and `tf.negative`.
* `tf.pack` and `tf.unpack` are deprecated in favor of `tf.stack` and
  `tf.unstack`.
* `TensorArray.pack` and `TensorArray.unpack` are getting deprecated in favor of
  `TensorArray.stack` and `TensorArray.unstack`.
* The following Python functions have had their arguments changed to use `axis`
  when referring to specific dimensions. We have kept the old keyword arguments
  for compatibility currently, but we will be removing them well before the
  final 1.0.
  * `tf.argmax`: `dimension` becomes `axis`
  * `tf.argmin`: `dimension` becomes `axis`
  * `tf.count_nonzero`: `reduction_indices` becomes `axis`
  * `tf.expand_dims`: `dim` becomes `axis`
  * `tf.reduce_all`: `reduction_indices` becomes `axis`
  * `tf.reduce_any`: `reduction_indices` becomes `axis`
  * `tf.reduce_join`: `reduction_indices` becomes `axis`
  * `tf.reduce_logsumexp`: `reduction_indices` becomes `axis`
  * `tf.reduce_max`: `reduction_indices` becomes `axis`
  * `tf.reduce_mean`: `reduction_indices` becomes `axis`
  * `tf.reduce_min`: `reduction_indices` becomes `axis`
  * `tf.reduce_prod`: `reduction_indices` becomes `axis`
  * `tf.reduce_sum`: `reduction_indices` becomes `axis`
  * `tf.reverse_sequence`: `batch_dim` becomes `batch_axis`, `seq_dim` becomes `seq_axis`
  * `tf.sparse_concat`: `concat_dim` becomes `axis`
  * `tf.sparse_reduce_sum`: `reduction_axes` becomes `axis`
  * `tf.sparse_reduce_sum_sparse`: `reduction_axes` becomes `axis`
  * `tf.sparse_split`: `split_dim` becomes `axis`
* `tf.listdiff` has been renamed to `tf.setdiff1d` to match NumPy naming.
* `tf.inv` has been renamed to be `tf.reciprocal` (component-wise reciprocal)
  to avoid confusion with `np.inv` which is matrix inversion
* tf.round now uses banker's rounding (round to even) semantics to match NumPy.
* `tf.split` now takes arguments in a reversed order and with different
  keywords. In particular, we now match NumPy order as
  `tf.split(value, num_or_size_splits, axis)`.
* `tf.sparse_split` now takes arguments in reversed order and with different
  keywords. In particular we now match NumPy order as
  `tf.sparse_split(sp_input, num_split, axis)`. NOTE: we have temporarily
  made `tf.sparse_split` require keyword arguments.
* `tf.concat` now takes arguments in reversed order and with different keywords. In particular we now match NumPy order as `tf.concat(values, axis, name)`.
* `tf.image.decode_jpeg` by default uses the faster DCT method, sacrificing
  a little fidelity for improved speed. One can revert to the old
  behavior by specifying the attribute `dct_method='INTEGER_ACCURATE'`.
* `tf.complex_abs` has been removed from the Python interface. `tf.abs`
  supports complex tensors and should be used instead.
* In the C++ API (in tensorflow/cc), Input, Output, etc. have moved
  from the tensorflow::ops namespace to tensorflow.
* Template.`var_scope` property renamed to `.variable_scope`
* SyncReplicasOptimizer is removed and SyncReplicasOptimizerV2 renamed to SyncReplicasOptimizer.
* `tf.zeros_initializer()` and `tf.ones_initializer()` now return a callable
  that must be called with initializer arguments, in your code replace
  `tf.zeros_initializer` with `tf.zeros_initializer()`.
* `SparseTensor.shape` has been renamed to `SparseTensor.dense_shape`.  Same for
  `SparseTensorValue.shape`.
* Replace tf.scalar_summary, tf.histogram_summary, tf.audio_summary, tf.image_summary with tf.summary.scalar, tf.summary.histogram, tf.summary.audio, tf.summary.image, respectively. The new summary ops take name rather than tag as their first argument, meaning summary ops now respect TensorFlow name scopes.
* Replace tf.train.SummaryWriter and tf.train.SummaryWriterCache with tf.summary.FileWriter and tf.summary.FileWriterCache.
* Removes RegisterShape from public API. Use C++ shape function registration
  instead.
* Deprecated `_ref` dtypes from the python API.
* In the C++ API (in tensorflow/cc), Input, Output, etc. have moved
  from the tensorflow::ops namespace to tensorflow.
* Change arg order for `{softmax,sparse_softmax,sigmoid}_cross_entropy_with_logits` to be (labels, predictions), and force use of named args.
* tf.nn.rnn_cell.* and most functions in tf.nn.rnn.* (with the exception of dynamic_rnn and raw_rnn) are temporarily in tf.contrib.rnn.  They will be moved back into core for TF 1.2.
* `tf.nn.sampled_softmax_loss` and `tf.nn.nce_loss` have both changed their API such that you need to switch the `inputs, labels` to `labels, inputs` parameters.
* The shape keyword argument of the `SparseTensor` constructor changes its name to `dense_shape` between Tensorflow 0.12 and Tensorflow 1.0.

## Bug Fixes and Other Changes
* Numerous C++ API updates.
* New op: `parallel_stack`.
* Introducing common tf io compression options constants for
  RecordReader/RecordWriter.
* Add `sparse_column_with_vocabulary_file`, to specify a feature column that
  transform string features to IDs, where the mapping is defined by a vocabulary
  file.
* Added `index_to_string_table` which returns a lookup table that maps indices to
  strings.
* Add `string_to_index_table`, which returns a lookup table that matches strings
  to indices.
* Add a `ParallelForWithWorkerId` function.
* Add `string_to_index_table`, which returns a lookup table that matches strings
  to indices.
* Support restore session from checkpoint files in v2 in `contrib/session_bundle`.
* Added a tf.contrib.image.rotate function for arbitrary angles.
* Added `tf.contrib.framework.filter_variables` as a convenience function to
  filter lists of variables based on regular expressions.
* `make_template()` takes an optional `custom_getter_ param`.
* Added comment about how existing directories are handled by
  `recursive_create_dir`.
* Added an op for QR factorizations.
* Divides and mods in Python API now use flooring (Python) semantics.
* Android: pre-built libs are now built nightly.
* Android: cmake/gradle build for TensorFlow Inference library under
  `contrib/android/cmake`
* Android: Much more robust Session initialization code.
* Android: TF stats now exposed directly in demo and log when debug mode is
  active
* Android: new/better README.md documentation
* saved_model is available as `tf.saved_model`.
* Empty op is now stateful.
* Improve speed of scatter_update on the cpu for ASSIGN operations.
* Change `reduce_join` to treat `reduction_indices` in the same way as other `reduce_` ops.
* Move `TensorForestEstimator` to `contrib/tensor_forest`.
* Enable compiler optimizations by default and allow configuration in configure.
* `tf.divide` now honors the name field.
* Make metrics weight broadcasting more strict.
* Add new queue-like `StagingArea` and new ops: `stage` and `unstage`.
* Enable inplace update ops for strings on CPU. Speed up string concat.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Aaron Hu, Abhishek Aggarwal, Adam Michael, Adriano Carmezim, @AfirSraftGarrier,
Alexander Novikov, Alexander Rosenberg Johansen, Andrew Gibiansky, Andrew Hundt,
Anish Shah, Anton Loss, @b0noI, @BoyuanJiang, Carl Thomé, Chad Kennedy, Comic
Chang, Connor Braa, Daniel N. Lang, Daniel Trebbien,
@danielgordon10, Darcy Liu, Darren Garvey, Dmitri Lapin, Eron Wright, Evan
Cofer, Fabrizio Milo, Finbarr Timbers, Franck Dernoncourt, Garrett Smith,
@guschmue, Hao Wei, Henrik Holst, Huazuo Gao, @Ian, @Issac, Jacob Israel,
Jangsoo Park, Jin Kim, Jingtian Peng, John Pope, Kye Bostelmann, Liangliang He,
Ling Zhang, Luheng He, Luke Iwanski, @lvli, Michael Basilyan, Mihir Patel,
Mikalai Drabovich, Morten Just, @newge, Nick Butlin, Nishant Shukla,
Pengfei Ni, Przemyslaw Tredak, @rasbt, @Ronny, Rudolf Rosa, @RustingSword,
Sam Abrahams, Sam Putnam, @SeongAhJo, Shi Jiaxin, @skavulya, Steffen MüLler,
@TheUSER123, @tiriplicamihai, @vhasanov, Victor Costan, Vit Stepanovs,
Wangda Tan, Wenjian Huang, Xingdong Zuo, Yaroslav Bulatov, Yota Toyama,
Yuan (Terry) Tang, Yuxin Wu

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


# Release 0.12.0

## Major Features and Improvements

* TensorFlow now builds and runs on Microsoft Windows (tested on Windows 10,
  Windows 7, and Windows Server 2016). Supported languages include Python (via a
  pip package) and C++. CUDA 8.0 and cuDNN 5.1 are supported for GPU
  acceleration. Known limitations include: It is not currently possible to load
  a custom op library. The GCS and HDFS file systems are not currently
  supported. The following ops are not currently implemented:
  Dequantize, QuantizeAndDequantize, QuantizedAvgPool,
  QuantizedBatchNomWithGlobalNormalization, QuantizedBiasAdd, QuantizedConcat,
  QuantizedConv2D, QuantizedMatmul, QuantizedMaxPool,
  QuantizeDownAndShrinkRange, QuantizedRelu, QuantizedRelu6, QuantizedReshape,
  QuantizeV2, RequantizationRange, and Requantize.
* Go: Experimental API in Go to create and execute graphs
  (https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)
* New checkpoint format becomes the default in `tf.train.Saver`. Old V1
  checkpoints continue to be readable; controlled by the `write_version`
  argument, `tf.train.Saver` now by default writes out in the new V2
  format. It significantly reduces the peak memory required and latency
  incurred during restore.
* Added a new library for library of matrix-free (iterative) solvers for linear
  equations, linear least-squares, eigenvalues and singular values in
  tensorflow/contrib/solvers. Initial version has lanczos bidiagonalization,
  conjugate gradients and CGLS.
* Added gradients for `matrix_solve_ls` and `self_adjoint_eig`.
* Large cleanup to add second order gradient for ops with C++ gradients and
  improve existing gradients such that most ops can now be differentiated
  multiple times.
* Added a solver for ordinary differential equations,
  `tf.contrib.integrate.odeint`.
* New contrib module for tensors with named axes, `tf.contrib.labeled_tensor`.
* Visualization of embeddings in TensorBoard.

## Breaking Changes to the API

* `BusAdjacency` enum replaced with a protocol buffer `DeviceLocality`.  PCI bus
  indexing now starts from 1 instead of 0, and `bus_id==0` is used where
  previously `BUS_ANY` was used.
* `Env::FileExists` and `FileSystem::FileExists` now return a tensorflow::Status
  instead of a bool. Any callers to this function can be converted to a bool
  by adding .ok() to the call.
* The C API type `TF_SessionWithGraph` has been renamed to `TF_Session`,
  indicating its preferred use in language bindings for TensorFlow.
  What was previously `TF_Session` has been renamed to `TF_DeprecatedSession`.
* Renamed `TF_Port` to `TF_Output` in the C API.
* Removes RegisterShape from public API. Use C++ shape function registration instead.
  indexing now starts from 1 instead of 0, and `bus_id==0` is used where
  previously `BUS_ANY` was used.
* Most RNN cells and RNN functions now use different variable scopes to be
  consistent with layers (`tf.contrib.layers`).  This means old checkpoints
  written using this code will not load after this change without providing
  `Saver` a list of variable renames.  Examples of variable scope changes
  include `RNN` -> `rnn` in `tf.nn.rnn`, `tf.nn.dynamic_rnn` and moving from
  `Linear/Matrix` -> `weights` and `Linear/Bias` -> `biases` in most RNN cells.
* Deprecated tf.select op. tf.where should be used instead.
* `SparseTensor.shape` has been renamed to `SparseTensor.dense_shape`.  Same for
  `SparseTensorValue.shape`.
* `Env::FileExists` and `FileSystem::FileExists` now return a
  `tensorflow::Status` instead of a bool. Any callers to this function can be
  converted to a bool by adding `.ok()` to the call.
* C API: Type `TF_SessionWithGraph` has been renamed to `TF_Session`, indicating
  its preferred use in language bindings for TensorFlow. What was previously
  `TF_Session` has been renamed to `TF_DeprecatedSession`.
* C API: Renamed `TF_Port` to `TF_Output`.
* C API: The caller retains ownership of `TF_Tensor` objects provided to
  `TF_Run`, `TF_SessionRun`, `TF_SetAttrTensor` etc.
* Renamed `tf.image.per_image_whitening()` to
  `tf.image.per_image_standardization()`
* Move Summary protobuf constructors to `tf.summary` submodule.
* Deprecate `histogram_summary`, `audio_summary`, `scalar_summary`,
  `image_summary`, `merge_summary`, and `merge_all_summaries`.
* Combined `batch_*` and regular version of linear algebra and FFT ops. The
  regular op now handles batches as well. All `batch_*` Python interfaces were
  removed.
* `tf.all_variables`, `tf.VARIABLES` and `tf.initialize_all_variables` renamed
  to `tf.global_variables`, `tf.GLOBAL_VARIABLES` and
  `tf.global_variables_initializer` respectively.
* `tf.zeros_initializer()` and `tf.ones_initializer()` now return a callable
  that must be called with initializer arguments, in your code replace
  `tf.zeros_initializer` with `tf.zeros_initializer()`

## Bug Fixes and Other Changes

* Use threadsafe version of `lgamma` function.
* Fix `tf.sqrt` handling of negative arguments.
* Fixed bug causing incorrect number of threads to be used for multi-threaded
  benchmarks.
* Performance optimizations for `batch_matmul` on multi-core CPUs.
* Improve trace, `matrix_set_diag`, `matrix_diag_part` and their gradients to
  work for rectangular matrices.
* Support for SVD of complex valued matrices.


## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

@a7744hsc, Abhi Agg, @admcrae, Adriano Carmezim, Aki Sukegawa, Alex Kendall,
Alexander Rosenberg Johansen, @amcrae, Amlan Kar, Andre Simpelo, Andreas Eberle,
Andrew Hundt, Arnaud Lenglet, @b0noI, Balachander Ramachandran, Ben Barsdell,
Ben Guidarelli, Benjamin Mularczyk, Burness Duan, @c0g, Changming Sun,
@chanis, Corey Wharton, Dan J, Daniel Trebbien, Darren Garvey, David Brailovsky,
David Jones, Di Zeng, @DjangoPeng, Dr. Kashif Rasul, @drag0, Fabrizio (Misto)
Milo, FabríCio Ceschin, @fp, @Ghedeon, @guschmue, Gökçen Eraslan, Haosdent
Huang, Haroen Viaene, Harold Cooper, Henrik Holst, @hoangmit, Ivan Ukhov, Javier
Dehesa, Jingtian Peng, Jithin Odattu, Joan Pastor, Johan Mathe, Johannes Mayer,
Jongwook Choi, Justus Schwabedal, Kai Wolf, Kamil Hryniewicz, Kamran Amini,
Karen Brems, Karl Lattimer, @kborer, Ken Shirriff, Kevin Rose, Larissa Laich,
Laurent Mazare, Leonard Lee, Liang-Chi Hsieh, Liangliang He, Luke Iwanski,
Marek Kolodziej, Moustafa Alzantot, @MrQianjinsi, @nagachika, Neil Han, Nick
Meehan, Niels Ole Salscheider, Nikhil Mishra, @nschuc, Ondrej Skopek, OndřEj
Filip, @OscarDPan, Pablo Moyano, Przemyslaw Tredak, @qitaishui, @Quarazy,
@raix852, Philipp Helo, Sam Abrahams, @SriramRamesh, Till Hoffmann, Tushar Soni,
@tvn, @tyfkda, Uwe Schmidt, Victor Villas, Vit Stepanovs, Vladislav Gubarev,
@wujingyue, Xuesong Yang, Yi Liu, Yilei Yang, @youyou3, Yuan (Terry) Tang,
Yuming Wang, Zafar Takhirov, @zhongyuk, Ziming Dong, @guotong1988

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.11.0

## Major Features and Improvements

* CUDA 8 support.
* cuDNN 5 support.
* HDFS Support.
* Adds Fused LSTM support via cuDNN 5 in `tensorflow/contrib/cudnn_rnn`.
* Improved support for NumPy style basic slicing including non-1 strides,
  ellipses, newaxis, and negative indices. For example complicated expressions
  like `foo[1, 2:4, tf.newaxis, ..., :-3:-1, :]` are now supported. In addition
  we have preliminary (non-broadcasting) support for sliced assignment to
  variables. In particular one can write `var[1:3].assign([1,11,111])`.
* Deprecated `tf.op_scope` and `tf.variable_op_scope` in favor of a unified `tf.name_scope` and `tf.variable_scope`. The new argument order of `tf.variable_scope` is incompatible with previous versions.
* Introducing `core/util/tensor_bundle` module: a module to efficiently
  serialize/deserialize tensors to disk.  Will be used in TF's new checkpoint
  format.
* Added tf.svd for computing the singular value decomposition (SVD) of dense
  matrices or batches of matrices (CPU only).
* Added gradients for eigenvalues and eigenvectors computed using
  `self_adjoint_eig` or `self_adjoint_eigvals`.
* Eliminated `batch_*` methods for most linear algebra and FFT ops and promoted
  the non-batch version of the ops to handle batches of matrices.
* Tracing/timeline support for distributed runtime (no GPU profiler yet).
* C API gives access to inferred shapes with `TF_GraphGetTensorNumDims` and
  `TF_GraphGetTensorShape`.
* Shape functions for core ops have moved to C++ via
  `REGISTER_OP(...).SetShapeFn(...)`.  Python shape inference RegisterShape calls
  use the C++ shape functions with `common_shapes.call_cpp_shape_fn`.  A future
  release will remove `RegisterShape` from python.


## Bug Fixes and Other Changes

* Documentation now includes operator overloads on Tensor and Variable.
* `tensorflow.__git_version__` now allows users to identify the version of the
  code that TensorFlow was compiled with. We also have
  `tensorflow.__git_compiler__` which identifies the compiler used to compile
  TensorFlow's core.
* Improved multi-threaded performance of `batch_matmul`.
* LSTMCell, BasicLSTMCell, and MultiRNNCell constructors now default to
  `state_is_tuple=True`.  For a quick fix while transitioning to the new
  default, simply pass the argument `state_is_tuple=False`.
* DeviceFactory's AddDevices and CreateDevices functions now return
  a Status instead of void.
* Int32 elements of list(type) arguments are no longer placed in host memory by
  default. If necessary, a list(type) argument to a kernel can be placed in host
  memory using a HostMemory annotation.
* `uniform_unit_scaling_initializer()` no longer takes a `full_shape` arg,
  instead relying on the partition info passed to the initializer function when
  it's called.
* The NodeDef protocol message is now defined in its own file `node_def.proto`
  `instead of graph.proto`.
* `ops.NoGradient` was renamed `ops.NotDifferentiable`. `ops.NoGradient` will
  be removed soon.
* `dot.h` / DotGraph was removed (it was an early analysis tool prior
  to TensorBoard, no longer that useful).  It remains in history
  should someone find the code useful.
* re2 / regexp.h was removed from being a public interface of TF.
  Should users need regular expressions, they should depend on the RE2
  library directly rather than via TensorFlow.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Abid K, @afshinrahimi, @AidanGG, Ajay Rao, Aki Sukegawa, Alex Rothberg,
Alexander Rosenberg Johansen, Andrew Gibiansky, Andrew Thomas, @Appleholic,
Bastiaan Quast, Ben Dilday, Bofu Chen, Brandon Amos, Bryon Gloden, Cissp®,
@chanis, Chenyang Liu, Corey Wharton, Daeyun Shin, Daniel Julius Lasiman, Daniel
Waterworth, Danijar Hafner, Darren Garvey, Denis Gorbachev, @DjangoPeng,
Egor-Krivov, Elia Palme, Eric Platon, Fabrizio Milo, Gaetan Semet,
Georg Nebehay, Gu Wang, Gustav Larsson, @haosdent, Harold Cooper, Hw-Zz,
@ichuang, Igor Babuschkin, Igor Macedo Quintanilha, Ilya Edrenkin, @ironhead,
Jakub Kolodziejczyk, Jennifer Guo, Jihun Choi, Jonas Rauber, Josh Bleecher
Snyder, @jpangburn, Jules Gagnon-Marchand, Karen Brems, @kborer, Kirill Bobyrev,
Laurent Mazare, Longqi Yang, Malith Yapa, Maniteja Nandana, Martin Englund,
Matthias Winkelmann, @mecab, Mu-Ik Jeon, Nand Dalal, Niels Ole Salscheider,
Nikhil Mishra, Park Jiin, Pieter De Rijk, @raix852, Ritwik Gupta, Sahil Sharma,
Sangheum Hwang, @SergejsRk, Shinichiro Hamaji, Simon Denel, @Steve, @suiyuan2009,
Tiago Jorge, Tijmen Tieleman, @tvn, @tyfkda, Wang Yang, Wei-Ting Kuo, Wenjian
Huang, Yan Chen, @YenChenLin, Yuan (Terry) Tang, Yuncheng Li, Yunfeng Wang, Zack
Polizzi, @zhongzyd, Ziming Dong, @perhapszzy

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.10.0

## Major Features and Improvements

* Added support for C++ shape inference
* Added graph-construction C API
* Major revision to the graph-construction C++ API
* Support makefile build for iOS
* Added Mac GPU support
* Full version of TF-Slim available as `tf.contrib.slim`
* Added k-Means clustering and WALS matrix factorization

## Bug Fixes and Other Changes

* Allow gradient computation for scalar values.
* Performance improvements for gRPC
* Improved support for fp16
* New high-level ops in tf.contrib.{layers,metrics}
* New features for TensorBoard, such as shape display, exponential smoothing
* Faster and more stable Google Cloud Storage (GCS) filesystem support
* Support for zlib compression and decompression for TFRecordReader and TFRecordWriter
* Support for reading (animated) GIFs
* Improved support for SparseTensor
* Added support for more probability distributions (Dirichlet, Beta, Bernoulli, etc.)
* Added Python interfaces to reset resource containers.
* Many bugfixes and performance improvements
* Many documentation fixes

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Alex Rothberg, Andrew Royer, Austin Marshall, @BlackCoal, Bob Adolf, Brian Diesel, Charles-Emmanuel Dias, @chemelnucfin, Chris Lesniewski, Daeyun Shin, Daniel Rodriguez, Danijar Hafner, Darcy Liu, Kristinn R. Thórisson, Daniel Castro, Dmitry Savintsev, Kashif Rasul, Dylan Paiton, Emmanuel T. Odeke, Ernest Grzybowski, Gavin Sherry, Gideon Dresdner, Gregory King, Harold Cooper, @heinzbeinz, Henry Saputra, Huarong Huo, Huazuo Gao, Igor Babuschkin, Igor Macedo Quintanilha, Ivan Ukhov, James Fysh, Jan Wilken Dörrie, Jihun Choi, Johnny Lim, Jonathan Raiman, Justin Francis, @lilac, Li Yi, Marc Khoury, Marco Marchesi, Max Melnick, Micael Carvalho, @mikowals, Mostafa Gazar, Nico Galoppo, Nishant Agrawal, Petr Janda, Yuncheng Li, @raix852, Robert Rose, @Robin-des-Bois, Rohit Girdhar, Sam Abrahams, satok16, Sergey Kishchenko, Sharkd Tu, @shotat, Siddharth Agrawal, Simon Denel, @sono-bfio, SunYeop Lee, Thijs Vogels, @tobegit3hub, @Undo1, Wang Yang, Wenjian Huang, Yaroslav Bulatov, Yuan Tang, Yunfeng Wang, Ziming Dong

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.9.0

## Major Features and Improvements

* Python 3.5 support and binaries
* Added iOS support
* Added support for processing on GPUs on MacOS
* Added makefile for better cross-platform build support (C API only)
* fp16 support and improved complex128 support for many ops
* Higher level functionality in contrib.{layers,losses,metrics,learn}
* More features to Tensorboard
* Improved support for string embedding and sparse features
* The RNN api is finally "official" (see, e.g., `tf.nn.dynamic_rnn`,
  `tf.nn.rnn`, and the classes in `tf.nn.rnn_cell`).
* TensorBoard now has an Audio Dashboard, with associated audio summaries.

## Bug Fixes and Other Changes

* Turned on CuDNN Autotune.
* Added support for using third-party Python optimization algorithms (contrib.opt).
* Google Cloud Storage filesystem support.
* HDF5 support
* Add support for 3d convolutions and pooling.
* Update gRPC release to 0.14.
* Eigen version upgrade.
* Switch to eigen thread pool
* `tf.nn.moments()` now accepts a `shift` argument. Shifting by a good estimate
  of the mean improves numerical stability. Also changes the behavior of the
  `shift` argument to `tf.nn.sufficient_statistics()`.
* Performance improvements
* Many bugfixes
* Many documentation fixes
* TensorBoard fixes: graphs with only one data point, Nan values,
  reload button and auto-reload, tooltips in scalar charts, run
  filtering, stable colors
* Tensorboard graph visualizer now supports run metadata. Clicking on nodes
  while viewing a stats for a particular run will show runtime statistics, such
  as memory or compute usage. Unused nodes will be faded out.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Aaron Schumacher, Aidan Dang, Akihiko ITOH, Aki Sukegawa, Arbit Chen, Aziz Alto, Danijar Hafner, Erik Erwitt, Fabrizio Milo, Felix Maximilian Möller, Henry Saputra, Sung Kim, Igor Babuschkin, Jan Zikes, Jeremy Barnes, Jesper Steen Møller, Johannes Mayer, Justin Harris, Kashif Rasul, Kevin Robinson, Loo Rong Jie, Lucas Moura, Łukasz Bieniasz-Krzywiec, Mario Cho, Maxim Grechkin, Michael Heilman, Mostafa Rahmani, Mourad Mourafiq, @ninotoshi, Orion Reblitz-Richardson, Yuncheng Li, @raoqiyu, Robert DiPietro, Sam Abrahams, Sebastian Raschka, Siddharth Agrawal, @snakecharmer1024, Stephen Roller, Sung Kim, SunYeop Lee, Thijs Vogels, Till Hoffmann, Victor Melo, Ville Kallioniemi, Waleed Abdulla, Wenjian Huang, Yaroslav Bulatov, Yeison Rodriguez, Yuan Tang, Yuxin Wu, @zhongzyd, Ziming Dong, Zohar Jackson

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

# Release 0.8.0

## Major Features and Improvements

* Added a distributed runtime using GRPC
* Move skflow to `contrib/learn`
* Better linear optimizer in `contrib/linear_optimizer`
* Random forest implementation in `contrib/tensor_forest`
* CTC loss and decoders in `contrib/ctc`
* Basic support for `half` data type
* Better support for loading user ops (see examples in `contrib/`)
* Allow use of (non-blocking) Eigen threadpool with `TENSORFLOW_USE_EIGEN_THREADPOOL` define
* Add an extension mechanism for adding network file system support
* TensorBoard displays metadata stats (running time, memory usage and device used) and tensor shapes

## Bug Fixes and Other Changes

* Utility for inspecting checkpoints
* Basic tracing and timeline support
* Allow building against cuDNN 5 (not incl. RNN/LSTM support)
* Added instructions and binaries for ProtoBuf library with fast serialization and without 64MB limit
* Added special functions
* `bool`-strictness: Tensors have to be explicitly compared to `None`
* Shape strictness: all fed values must have a shape that is compatible with the tensor they are replacing
* Exposed `tf.while_loop` (deprecated `control_flow_ops.While`)
* run() now takes RunOptions and RunMetadata, which enable timing stats
* Fixed lots of potential overflow problems in op kernels
* Various performance improvements, especially for RNNs and convolutions
* Many bugfixes
* Nightly builds, tutorial tests, many test improvements
* New examples: transfer learning and deepdream ipython notebook
* Added tutorials, many documentation fixes.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Abhinav Upadhyay, Aggelos Avgerinos, Alan Wu, Alexander G. de G. Matthews, Aleksandr Yahnev, @amchercashin, Andy Kitchen, Aurelien Geron, Awni Hannun, @BanditCat, Bas Veeling, Cameron Chen, @cg31, Cheng-Lung Sung, Christopher Bonnett, Dan Becker, Dan Van Boxel, Daniel Golden, Danijar Hafner, Danny Goodman, Dave Decker, David Dao, David Kretch, Dongjoon Hyun, Dustin Dorroh, @e-lin, Eurico Doirado, Erik Erwitt, Fabrizio Milo, @gaohuazuo, Iblis Lin, Igor Babuschkin, Isaac Hodes, Isaac Turner, Iván Vallés, J Yegerlehner, Jack Zhang, James Wexler, Jan Zikes, Jay Young, Jeff Hodges, @jmtatsch, Johnny Lim, Jonas Meinertz Hansen, Kanit Wongsuphasawat, Kashif Rasul, Ken Shirriff, Kenneth Mitchner, Kenta Yonekura, Konrad Magnusson, Konstantin Lopuhin, @lahwran, @lekaha, @liyongsea, Lucas Adams, @makseq, Mandeep Singh, @manipopopo, Mark Amery, Memo Akten, Michael Heilman, Michael Peteuil, Nathan Daly, Nicolas Fauchereau, @ninotoshi, Olav Nymoen, @panmari, @papelita1234, Pedro Lopes, Pranav Sailesh Mani, RJ Ryan, Rob Culliton, Robert DiPietro, @ronrest, Sam Abrahams, Sarath Shekkizhar, Scott Graham, Sebastian Raschka, Sung Kim, Surya Bhupatiraju, Syed Ahmed, Till Hoffmann, @timsl, @urimend, @vesnica, Vlad Frolov, Vlad Zagorodniy, Wei-Ting Kuo, Wenjian Huang, William Dmitri Breaden Madden, Wladimir Schmidt, Yuan Tang, Yuwen Yan, Yuxin Wu, Yuya Kusakabe, @zhongzyd, @znah.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


# Release 0.7.1

## Bug Fixes and Other Changes

* Added gfile.Open and gfile.Copy, used by input_data.py.
* Fixed Saver bug when MakeDirs tried to create empty directory.
* GPU Pip wheels are built with cuda 7.5 and cudnn-v4, making them
  required for the binary releases. Lower versions of cuda/cudnn can
  be supported by installing from sources and setting the options
  during ./configure
* Fix dataset encoding example for Python3 (@danijar)
* Fix PIP installation by not packaging protobuf as part of wheel,
  require protobuf 3.0.0b2.
* Fix Mac pip installation of numpy by requiring pip >= 1.10.1.
* Improvements and fixes to Docker image.


# Release 0.7.0

## Major Features and Improvements

* Allow using any installed Cuda >= 7.0 and cuDNN >= R2, and add support
  for cuDNN R4
* Added a `contrib/` directory for unsupported or experimental features,
  including higher level `layers` module
* Added an easy way to add and dynamically load user-defined ops
* Built out a good suite of tests, things should break less!
* Added `MetaGraphDef` which makes it easier to save graphs with metadata
* Added assignments for "Deep Learning with TensorFlow" udacity course


## Bug Fixes and Other Changes

* Added a versioning framework for `GraphDef`s to ensure compatibility
* Enforced Python 3 compatibility
* Internal changes now show up as sensibly separated commits
* Open-sourced the doc generator
* Un-fork Eigen
* Simplified the `BUILD` files and cleaned up C++ headers
* TensorFlow can now be used as a submodule in another bazel build
* New ops (e.g., `*fft`, `*_matrix_solve`)
* Support for more data types in many ops
* Performance improvements
* Various bugfixes
* Documentation fixes and improvements


## Breaking Changes to the API

* `AdjustContrast` kernel deprecated, new kernel `AdjustContrastv2` takes and
  outputs float only. `adjust_contrast` now takes all data types.
* `adjust_brightness`'s `delta` argument is now always assumed to be in `[0,1]`
  (as is the norm for images in floating point formats), independent of the
  data type of the input image.
* The image processing ops do not take `min` and `max` inputs any more, casting
  safety is handled by `saturate_cast`, which makes sure over- and underflows
  are handled before casting to data types with smaller ranges.
* For C++ API users: `IsLegacyScalar` and `IsLegacyVector` are now gone from
  `TensorShapeUtils` since TensorFlow is scalar strict within Google (for
  example, the shape argument to `tf.reshape` can't be a scalar anymore).  The
  open source release was already scalar strict, so outside Google `IsScalar`
  and `IsVector` are exact replacements.
* The following files are being removed from `tensorflow/core/public/`:
    * `env.h` -> `../platform/env.h`
    * `status.h` -> `../lib/core/status.h`
    * `tensor.h` -> `../framework/tensor.h`
    * `tensor_shape.h` -> `../framework/tensor_shape.h`
    * `partial_tensor_shape.h` -> `../framework/partial_tensor_shape.h`
    * `tensorflow_server.h` deleted
* For C++ API users: `TensorShape::ShortDebugString` has been renamed to
  `DebugString`, and the previous `DebugString` behavior is gone (it was
  needlessly verbose and produced a confusing empty string for scalars).
* `GraphOptions.skip_common_subexpression_elimination` has been removed. All
  graph optimizer options are now specified via
  `GraphOptions.OptimizerOptions`.
* `ASSERT_OK` / `EXPECT_OK` macros conflicted with external projects, so they
  were renamed `TF_ASSERT_OK`, `TF_EXPECT_OK`.  The existing macros are
  currently maintained for short-term compatibility but will be removed.
* The non-public `nn.rnn` and the various `nn.seq2seq` methods now return
  just the final state instead of the list of all states.
* `tf.scatter_update` now no longer guarantees that lexicographically largest
  index be used for update when duplicate entries exist.
* `tf.image.random_crop(image, [height, width])` is now
  `tf.random_crop(image, [height, width, depth])`, and `tf.random_crop` works
  for any rank (not just 3-D images).  The C++ `RandomCrop` op has been replaced
  with pure Python.
* Renamed `tf.test.GetTempDir` and `tf.test.IsBuiltWithCuda` to
  `tf.test.get_temp_dir` and `tf.test.is_built_with_cuda` for PEP-8
  compatibility.
* `parse_example`'s interface has changed, the old interface is accessible in
  `legacy_parse_example` (same for related functions).
* New `Variable`s are not added to the same collection several times even if
  a list with duplicates is passed to the constructor.
* The Python API will now properly set the `list` member of `AttrValue` in
  constructed `GraphDef` messages for empty lists.  The serialization of some
  graphs will change, but the change is both forwards and backwards compatible.
  It will break tests that compare a generated `GraphDef` to a golden serialized
  `GraphDef` (which is discouraged).


## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Akiomi Kamakura, Alex Vig, Alexander Rosenberg Johansen, Andre Cruz, Arun Ahuja,
Bart Coppens, Bernardo Pires, Carl Vondrick, Cesar Salgado, Chen Yu,
Christian Jauvin, Damien Aymeric, Dan Vanderkam, Denny Britz, Dongjoon Hyun,
Eren Güven, Erik Erwitt, Fabrizio Milo, G. Hussain Chinoy, Jim Fleming,
Joao Felipe Santos, Jonas Meinertz Hansen, Joshi Rekha, Julian Viereck,
Keiji Ariyama, Kenton Lee, Krishna Sankar, Kristina Chodorow, Linchao Zhu,
Lukas Krecan, Mark Borgerding, Mark Daoust, Moussa Taifi,
Nathan Howell, Naveen Sundar Govindarajulu, Nick Sweeting, Niklas Riekenbrauck,
Olivier Grisel, Patrick Christ, Povilas Liubauskas, Rainer Wasserfuhr,
Romain Thouvenin, Sagan Bolliger, Sam Abrahams, Taehoon Kim, Timothy J Laurent,
Vlad Zavidovych, Yangqing Jia, Yi-Lin Juang, Yuxin Wu, Zachary Lipton,
Zero Chen, Alan Wu, @brchiu, @emmjaykay, @jalammar, @Mandar-Shinde,
@nsipplswezey, @ninotoshi, @panmari, @prolearner and @rizzomichaelg.

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.


# Release 0.6.0

## Major Features and Improvements

* Python 3.3+ support via changes to python codebase and ability
  to specify python version via ./configure.

* Some improvements to GPU performance and memory usage:
  [convnet benchmarks](https://github.com/soumith/convnet-benchmarks/issues/66)
  roughly equivalent with native cudnn v2 performance.  Improvements mostly due
  to moving to 32-bit indices, faster shuffling kernels.  More improvements to
  come in later releases.


## Bug Fixes

* Lots of fixes to documentation and tutorials, many contributed
  by the public.

* 271 closed issues on github issues.

## Backwards-Incompatible Changes

* `tf.nn.fixed_unigram_candidate_sampler` changed its default 'distortion'
  attribute from 0.0 to 1.0. This was a bug in the original release
  that is now fixed.

# Release 0.5.0

Initial release of TensorFlow.
