<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4724125.svg)](https://doi.org/10.5281/zenodo.4724125)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow/badge)](https://securityscorecards.dev/viewer/?uri=github.com/tensorflow/tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow-py.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow-py)
[![OSSRank](https://shields.io/endpoint?url=https://ossrank.com/shield/44)](https://ossrank.com/p/44)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![TF Official Continuous](https://tensorflow.github.io/build/TF%20Official%20Continuous.svg)](https://tensorflow.github.io/build#TF%20Official%20Continuous)
[![TF Official Nightly](https://tensorflow.github.io/build/TF%20Official%20Nightly.svg)](https://tensorflow.github.io/build#TF%20Official%20Nightly)

**`Documentación`** |
--------------------- |
[![Documentación](https://img.shields.io/badge/api-referencia-blue.svg)](https://www.tensorflow.org/api_docs/) |

[TensorFlow](https://www.tensorflow.org/) es una plataforma de código abierto de extremo a extremo para aprendizaje automático. Tiene un ecosistema completo y flexible de [herramientas](https://www.tensorflow.org/resources/tools), [bibliotecas](https://www.tensorflow.org/resources/libraries-extensions) y recursos de [comunidad](https://www.tensorflow.org/community) que permiten a los investigadores avanzar en el estado del arte en ML y a los desarrolladores construir y desplegar fácilmente aplicaciones potenciadas por ML.

TensorFlow fue desarrollado originalmente por investigadores e ingenieros que trabajaban en el equipo de Inteligencia Artificial en Google Brain para llevar a cabo investigaciones en aprendizaje automático y redes neuronales. Sin embargo, el framework es lo suficientemente versátil como para ser utilizado en otras áreas también.

TensorFlow proporciona APIs estables en [Python](https://www.tensorflow.org/api_docs/python) y [C++](https://www.tensorflow.org/api_docs/cc), así como una API compatible con versiones anteriores pero no garantizada para [otros lenguajes](https://www.tensorflow.org/api_docs).

Mantente actualizado con los anuncios de lanzamiento y las actualizaciones de seguridad suscribiéndote a [announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce). Consulta todas las [listas de correo](https://www.tensorflow.org/community/forums).

## Instalación

Consulta la [guía de instalación de TensorFlow](https://www.tensorflow.org/install) para obtener el [paquete pip](https://www.tensorflow.org/install/pip), habilitar el soporte de GPU, usar un [contenedor Docker](https://www.tensorflow.org/install/docker) y [compilar desde el código fuente](https://www.tensorflow.org/install/source).

Para instalar la versión actual, que incluye soporte para tarjetas GPU habilitadas para CUDA *(Ubuntu y Windows)*:



Otros dispositivos (DirectX y MacOS-metal) son compatibles mediante [plugins de dispositivos](https://www.tensorflow.org/install/gpu_plugins#available_devices).

También está disponible un paquete más pequeño solo para CPU:


Para actualizar TensorFlow a la última versión, agrega el indicador `--upgrade` a los comandos anteriores.

*Binarios nocturnos están disponibles para pruebas utilizando los paquetes [tf-nightly](https://pypi.python.org/pypi/tf-nightly) y [tf-nightly-cpu](https://pypi.python.org/pypi/tf-nightly-cpu) en PyPi.*

#### *Prueba tu primer programa de TensorFlow*

```shell
$ python
>>> import tensorflow as tf
>>> tf.add(1, 2).numpy()
3
>>> hello = tf.constant('¡Hola, TensorFlow!')
>>> hello.numpy()
b'¡Hola, TensorFlow!'


```
Para más ejemplos, consulta los tutoriales de TensorFlow.

## Directrices de contribución
Si deseas contribuir a TensorFlow, asegúrate de revisar las directrices de contribución. Este proyecto se adhiere al código de conducta de TensorFlow. Al participar, se espera que sigas este código.

Utilizamos problemas de GitHub para hacer seguimiento de solicitudes y errores. Por favor, consulta el Foro de TensorFlow para preguntas generales y discusión, y por favor dirige preguntas específicas a Stack Overflow.

El proyecto TensorFlow se esfuerza por cumplir con las prácticas recomendadas generalmente aceptadas en el desarrollo de software de código abierto.

## Directrices de parches
Sigue estos pasos para parchear una versión específica de TensorFlow, por ejemplo, para aplicar correcciones a errores o vulnerabilidades de seguridad:

Clona el repositorio de TensorFlow y cambia al branch correspondiente para tu versión deseada de TensorFlow, por ejemplo, el branch r2.8 para la versión 2.8.
Aplica (es decir, cherry-pick) los cambios deseados y resuelve cualquier conflicto de código.
Ejecuta las pruebas de TensorFlow y asegúrate de que pasen




