<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4724125.svg)](https://doi.org/10.5281/zenodo.4724125)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow/badge)](https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow-py.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow-py)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

**`Documentación`** |
------------------- |
[![Documentación](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

[TensorFlow](https://www.tensorflow.org/) TensorFlow es una plataforma de código abierto integral para el aprendizaje automático. Tiene un ecosistema integral 
y flexible de
[herramientas](https://www.tensorflow.org/resources/tools),
[librerias](https://www.tensorflow.org/resources/libraries-extensions), y recursos
[comunitarios](https://www.tensorflow.org/community) que permite a los investigadores impulsar lo último en ML 
y a los desarrolladores crear e implementar fácilmente aplicaciones basadas en ML.

TensorFlow fue desarrollado originalmente por investigadores 
e ingenieros que trabajaban en el equipo de Google Brain dentro 
de la organización de investigación de inteligencia de máquinas de 
Google para llevar a cabo investigaciones sobre aprendizaje automático y 
redes neuronales profundas. El sistema es lo suficientemente general 
como para ser aplicable también en una amplia variedad de otros dominios.

TensorFlow proporciona APIs estables de [Python](https://www.tensorflow.org/api_docs/python)
y [C++](https://www.tensorflow.org/api_docs/cc), así como una API compatible con versiones anteriores 
no garantizada para [otros idiomas](https://www.tensorflow.org/api_docs).

Manténgase actualizado con los anuncios de lanzamiento y las actualizaciones de seguridad suscribiéndose a
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).
Ver todas las [listas de correo](https://www.tensorflow.org/community/forums).

## Instalación
Consulte la [guía de instalación de TensorFlow](https://www.tensorflow.org/install) para el
[paquete pip](https://www.tensorflow.org/install/pip), para
[habilitar la compatibilidad con GPU](https://www.tensorflow.org/install/gpu), usar un
[contenedor Docker](https://www.tensorflow.org/install/docker), y
[compilar desde el origen](https://www.tensorflow.org/install/source).

Para instalar la versión actual, que incluye soporte para 
[tarjetas GPU habilitadas para CUDA](https://www.tensorflow.org/install/gpu) *(Ubuntu y
Windows)*:

```
$ pip install tensorflow
```

También está disponible un paquete más pequeño solo para CPU:

```
$ pip install tensorflow-cpu
```

Para actualizar TensorFlow a la última versión, agregue el indicador `--upgrade` a los anteriores
comandos

*Los binarios nocturnos están disponibles para probar usando el
[tf-nightly](https://pypi.python.org/pypi/tf-nightly) y
[tf-nightly-cpu](https://pypi.python.org/pypi/tf-nightly-cpu) paquetes en PyPi.*

#### *Pruebs tu primer programa TensorFlow*

```shell
$ python
```

```python
>>> import tensorflow as tf
>>> tf.add(1, 2).numpy()
3
>>> hello = tf.constant('Hello, TensorFlow!')
>>> hello.numpy()
b'Hello, TensorFlow!'
```

Para más ejemplos, mira los
[tutoriales de Tensorflow](https://www.tensorflow.org/tutorials/).

## Pautas de contribución

**Si desea contribuir con TensorFlow, asegúrese de revisar las
[pautas de contribución](CONTRIBUTING.md). Este proyecto se adhiere a el
[código de conducta](CODE_OF_CONDUCT.md) de Tensorflow. Al participar, se espera que usted
mantenga este código.**

**Nosotros usamos [GitHub issues](https://github.com/tensorflow/tensorflow/issues) para
solicitudes de seguimiento y errores, consulte
[TensorFlow Discuss](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)
para preguntas generales y discusión, y por favor dirija preguntas específicas a
[Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).**

El proyecto TensorFlow se esfuerza por cumplir con las mejores prácticas generalmente aceptadas en
desarrollo de software de código abierto.

## Estado de construcción continuo

Puede encontrar más plataformas y configuraciones compatibles con la comunidad en el
[TensorFlow SIG Build community builds table](https://github.com/tensorflow/build#community-supported-tensorflow-builds).

### Builds Oficiales

Build Type                    | Status                                                                                                                                                                           | Artifacts
----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
**Linux CPU**                 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.html)           | [PyPI](https://pypi.org/project/tf-nightly/)
**Linux GPU**                 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.html) | [PyPI](https://pypi.org/project/tf-nightly-gpu/)
**Linux XLA**                 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.html)         | TBA
**macOS**                     | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.html)     | [PyPI](https://pypi.org/project/tf-nightly/)
**Windows CPU**               | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.html)       | [PyPI](https://pypi.org/project/tf-nightly/)
**Windows GPU**               | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.html)       | [PyPI](https://pypi.org/project/tf-nightly-gpu/)
**Android**                   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.html)               | [Download](https://bintray.com/google/tensorflow/tensorflow/_latestVersion)
**Raspberry Pi 0 and 1**      | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.html)           | [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv6l.whl)
**Raspberry Pi 2 and 3**      | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.html)           | [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv7l.whl)
**Libtensorflow MacOS CPU**   | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/macos/latest/macos_cpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Linux CPU**   | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/ubuntu_16/latest/cpu/ubuntu_cpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Linux GPU**   | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/ubuntu_16/latest/gpu/ubuntu_gpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Windows CPU** | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/windows/latest/cpu/windows_cpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)
**Libtensorflow Windows GPU** | Status Temporarily Unavailable                                                                                                                                                   | [Nightly Binary](https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/windows/latest/gpu/windows_gpu_libtensorflow_binaries.tar.gz) [Official GCS](https://storage.googleapis.com/tensorflow/)

## Recursos

*   [TensorFlow.org](https://www.tensorflow.org)
*   [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/)
*   [TensorFlow Official Models](https://github.com/tensorflow/models/tree/master/official)
*   [TensorFlow Examples](https://github.com/tensorflow/examples)
*   [DeepLearning.AI TensorFlow Developer Professional Certificate](https://www.coursera.org/specializations/tensorflow-in-practice)
*   [TensorFlow: Data and Deployment from Coursera](https://www.coursera.org/specializations/tensorflow-data-and-deployment)
*   [Getting Started with TensorFlow 2 from Coursera](https://www.coursera.org/learn/getting-started-with-tensor-flow2)
*   [TensorFlow: Advanced Techniques from Coursera](https://www.coursera.org/specializations/tensorflow-advanced-techniques)
*   [TensorFlow 2 for Deep Learning Specialization from Coursera](https://www.coursera.org/specializations/tensorflow2-deeplearning)
*   [Intro to TensorFlow for A.I, M.L, and D.L from Coursera](https://www.coursera.org/learn/introduction-tensorflow)
*   [Intro to TensorFlow for Deep Learning from Udacity](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
*   [Introduction to TensorFlow Lite from Udacity](https://www.udacity.com/course/intro-to-tensorflow-lite--ud190)
*   [Machine Learning with TensorFlow on GCP](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp)
*   [TensorFlow Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)
*   [TensorFlow Blog](https://blog.tensorflow.org)
*   [Learn ML with TensorFlow](https://www.tensorflow.org/resources/learn-ml)
*   [TensorFlow Twitter](https://twitter.com/tensorflow)
*   [TensorFlow YouTube](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)
*   [TensorFlow model optimization roadmap](https://www.tensorflow.org/model_optimization/guide/roadmap)
*   [TensorFlow White Papers](https://www.tensorflow.org/about/bib)
*   [TensorBoard Visualization Toolkit](https://github.com/tensorflow/tensorboard)
*   [TensorFlow Code Search](https://cs.opensource.google/tensorflow/tensorflow)

Aprenda más sobre la
[comunidad TensorFlow](https://www.tensorflow.org/community) y como
[contribuir](https://www.tensorflow.org/community/contribute).

## Licencia

[Apache License 2.0](LICENSE)
