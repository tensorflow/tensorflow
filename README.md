<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png" alt="TensorFlow Logo" style="max-width: 100%; height: auto;">
</div>

<div align="center">
  <a href="https://badge.fury.io/py/tensorflow">
    <img src="https://img.shields.io/pypi/pyversions/tensorflow.svg" alt="Python Version">
  </a>
  <a href="https://badge.fury.io/py/tensorflow">
    <img src="https://badge.fury.io/py/tensorflow.svg" alt="PyPI">
  </a>
  <a href="https://doi.org/10.5281/zenodo.4724125">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4724125.svg" alt="DOI">
  </a>
  <a href="https://bestpractices.coreinfrastructure.org/projects/1486">
    <img src="https://bestpractices.coreinfrastructure.org/projects/1486/badge" alt="CII Best Practices">
  </a>
  <a href="https://securityscorecards.dev/viewer/?uri=github.com/tensorflow/tensorflow">
    <img src="https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow/badge" alt="OpenSSF Scorecard">
  </a>
  <a href="https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow">
    <img src="https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg" alt="Fuzzing Status">
  </a>
  <a href="https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow-py">
    <img src="https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow-py.svg" alt="Fuzzing Status">
  </a>
  <a href="https://ossrank.com/p/44">
    <img src="https://shields.io/endpoint?url=https://ossrank.com/shield/44" alt="OSSRank">
  </a>
  <a href="CODE_OF_CONDUCT.md">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg" alt="Contributor Covenant">
  </a>
  <a href="https://tensorflow.github.io/build#TF%20Official%20Continuous">
    <img src="https://tensorflow.github.io/build/TF%20Official%20Continuous.svg" alt="TF Official Continuous">
  </a>
  <a href="https://tensorflow.github.io/build#TF%20Official%20Nightly">
    <img src="https://tensorflow.github.io/build/TF%20Official%20Nightly.svg" alt="TF Official Nightly">
  </a>
</div>

<h3 align="center">Documentation</h3>
<p align="center">
  <a href="https://www.tensorflow.org/api_docs/">
    <img src="https://img.shields.io/badge/api-reference-blue.svg" alt="Documentation">
  </a>
</p>

<p align="justify">
  <strong><a href="https://www.tensorflow.org/">TensorFlow</a></strong> is an end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of 
  <a href="https://www.tensorflow.org/resources/tools">tools</a>, 
  <a href="https://www.tensorflow.org/resources/libraries-extensions">libraries</a>, and 
  <a href="https://www.tensorflow.org/community">community</a> resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.
</p>

<p align="justify">
  TensorFlow was originally developed by researchers and engineers working within the Machine Intelligence team at Google Brain to conduct research in machine learning and neural networks. However, the framework is versatile enough to be used in other areas as well.
</p>

<p align="justify">
  TensorFlow provides stable 
  <a href="https://www.tensorflow.org/api_docs/python">Python</a> and 
  <a href="https://www.tensorflow.org/api_docs/cc">C++</a> APIs, as well as a non-guaranteed backward compatible API for 
  <a href="https://www.tensorflow.org/api_docs">other languages</a>.
</p>

<p align="justify">
  Keep up-to-date with release announcements and security updates by subscribing to 
  <a href="https://groups.google.com/a/tensorflow.org/forum/#!forum/announce">announce@tensorflow.org</a>. See all the 
  <a href="https://www.tensorflow.org/community/forums">mailing lists</a>.
</p>

<h2>Install</h2>
<p align="justify">
  See the <a href="https://www.tensorflow.org/install">TensorFlow install guide</a> for the 
  <a href="https://www.tensorflow.org/install/pip">pip package</a>, to 
  <a href="https://www.tensorflow.org/install/gpu">enable GPU support</a>, use a 
  <a href="https://www.tensorflow.org/install/docker">Docker container</a>, and 
  <a href="https://www.tensorflow.org/install/source">build from source</a>.
</p>

<p align="justify">
  To install the current release, which includes support for 
  <a href="https://www.tensorflow.org/install/gpu">CUDA-enabled GPU cards</a> (Ubuntu and Windows):
</p>

<pre><code>$ pip install tensorflow</code></pre>

<p align="justify">
  Other devices (DirectX and MacOS-metal) are supported using 
  <a href="https://www.tensorflow.org/install/gpu_plugins#available_devices">Device plugins</a>.
</p>

<p align="justify">
  A smaller CPU-only package is also available:
</p>

<pre><code>$ pip install tensorflow-cpu</code></pre>

<p align="justify">
  To update TensorFlow to the latest version, add <code>--upgrade</code> flag to the above commands.
</p>

<p align="justify">
  Nightly binaries are available for testing using the 
  <a href="https://pypi.python.org/pypi/tf-nightly">tf-nightly</a> and 
  <a href="https://pypi.python.org/pypi/tf-nightly-cpu">tf-nightly-cpu</a> packages on PyPi.
</p>

<h4>Try your first TensorFlow program</h4>

<pre><code>$ python
>>> import tensorflow as tf
>>> tf.add(1, 2).numpy()
3
>>> hello = tf.constant('Hello, TensorFlow!')
>>> hello.numpy()
b'Hello, TensorFlow!'
</code></pre>

<p align="justify">
  For more examples, see the <a href="https://www.tensorflow.org/tutorials/">TensorFlow tutorials</a>.
</p>

<h2>Contribution Guidelines</h2>
<p align="justify">
  If you want to contribute to TensorFlow, be sure to review the 
  <a href="CONTRIBUTING.md">contribution guidelines</a>. This project adheres to TensorFlow's 
  <a href="CODE_OF_CONDUCT.md">code of conduct</a>. By participating, you are expected to uphold this code.
</p>

<p align="justify">
  We use <a href="https://github.com/tensorflow/tensorflow/issues">GitHub issues</a> for tracking requests and bugs. Please see the 
  <a href="https://discuss.tensorflow.org/">TensorFlow Forum</a> for general questions and discussion, and please direct specific questions to 
  <a href="https://stackoverflow.com/questions/tagged/tensorflow">Stack Overflow</a>.
</p>

<p align="justify">
  The TensorFlow project strives to abide by generally accepted best practices in open-source software development.
</p>

<h2>Patching Guidelines</h2>
<p align="justify">
  Follow these steps to patch a specific version of TensorFlow, for example, to apply fixes to bugs or security vulnerabilities:
</p>
<ul>
  <li>Clone the TensorFlow repo and switch to the corresponding branch for your desired TensorFlow version, for example, branch <code>r2.8</code> for version 2.8.</li>
  <li>Apply (that is, cherry-pick) the desired changes and resolve any code conflicts.</li>
  <li>Run TensorFlow tests and ensure they pass.</li>
  <li><a href="https://www.tensorflow.org/install/source">Build</a> the TensorFlow pip package from source.</li>
</ul>

<h2>Continuous Build Status</h2>
<p align="justify">
  You can find more community-supported platforms and configurations in the 
  <a href="https://github.com/tensorflow/build#community-supported-tensorflow-builds">TensorFlow SIG Build community builds table</a>.
</p>

<h3>Official Builds</h3>
<table>
  <thead>
    <tr>
      <th>Platform</th>
      <th>Builds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Windows</td>
      <td><a href="https://www.tensorflow.org/install/pip#windows">Install</a></td>
    </tr>
    <tr>
      <td>Linux</td>
      <td><a href="https://www.tensorflow.org/install/pip#linux">Install</a></td>
    </tr>
    <tr>
      <td>macOS</td>
      <td><a href="https://www.tensorflow.org/install/pip#macos">Install</a></td>
    </tr>
  </tbody>
</table>

<p align="justify">
  For more information, please visit the <a href="https://www.tensorflow.org/community/contribute">TensorFlow Community page</a>.
</p>
