<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4724125.svg)](https://doi.org/10.5281/zenodo.4724125)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow/badge)](https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow-py.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow-py)
[![OSSRank](https://shields.io/endpoint?url=https://ossrank.com/shield/44)](https://ossrank.com/p/44)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![TF Official Continuous](https://tensorflow.github.io/build/TF%20Official%20Continuous.svg)](https://tensorflow.github.io/build#TF%20Official%20Continuous)
[![TF Official Nightly](https://tensorflow.github.io/build/TF%20Official%20Nightly.svg)](https://tensorflow.github.io/build#TF%20Official%20Nightly)

**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

[TensorFlow](https://www.tensorflow.org/) est une plateforme open source de bout en bout 
pour l'apprentissage automatique.  Il dispose d'un écosystème complet et flexible d'
[outils](https://www.tensorflow.org/resources/tools), de
[bibliotheques](https://www.tensorflow.org/resources/libraries-extensions), et d'une
[communauté](https://www.tensorflow.org/community) de resources qui permet aux chercheurs
aux chercheurs de faire avancer l'état de l'art en matière de ML et aux développeurs
de créer et de déployer facilement des applications basées sur la ML.

TensorFlow a été développé à l'origine par des chercheurs et des ingénieurs travaillant
dans l'équipe Google Brain au sein de l'organisation Machine Intelligence Research de
Google pour mener des recherches sur le "machine learning" et "deep neural networks research". 
Le système est suffisamment général pour être également applicable dans une grande variété 
d'autres domaines.

TensorFlow fournit des API [Python](https://www.tensorflow.org/api_docs/python)
et [C++](https://www.tensorflow.org/api_docs/cc) stables, ainsi qu'une 
rétrocompatibilité non garantie API pour [autre langage](https://www.tensorflow.org/api_docs).

Tenez-vous au courant des annonces de publication et des mises à jour de securité 
en vous abonnant à
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).
Voir toutes les [listes de diffusion](https://www.tensorflow.org/community/forums).

## Installation

Consultez [guide d'installation de TensorFlow](https://www.tensorflow.org/install) pour
[pip package](https://www.tensorflow.org/install/pip), pour
[activer la prise en charge GPU](https://www.tensorflow.org/install/gpu), utilisez un
[Conteneur Docker](https://www.tensorflow.org/install/docker), et
[construire à partir de la source](https://www.tensorflow.org/install/source).

Pour installer la version actuelle, qui inclut la prise en charge de
[Carte GPU compatibles CUDA](https://www.tensorflow.org/install/gpu) *(Ubuntu et
Windows)*:

```
$ pip install tensorflow
```

D'autres périphériques (DirectX and MacOS-metal) sont pris en charge
à l'aide de [Plugins de périphériques]
(https://www.tensorflow.org/install/gpu_plugins#available_devices).

Un ensemble plus petit, composé uniquement de l'unité centrale, est
également disponible:

```
$ pip install tensorflow-cpu
```

Pour mettre à jour TensorFlow vers la dernière version, ajoutez le drapeau `--upgrade` 
aux commandes ci-dessus.

*Des binaires nocturnes sont disponibles pour être testés en utilisant les fichiers
[tf-nightly](https://pypi.python.org/pypi/tf-nightly) et
[tf-nightly-cpu](https://pypi.python.org/pypi/tf-nightly-cpu) packages on PyPi.*

#### *Essayer votre premier programme TensorFlow*

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

Pour plus d'exemples, consultez le
[TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Directives de contributions

**Si vous souhaitez contribuer à TensorFlow, assurez-vous de lire les
[directives de contribution](CONTRIBUTING.md). Ce projet adhère au
[code de conduite](CODE_OF_CONDUCT.md) de Tensorflow. En participant à 
ce projet, vous vous engagez à respecter ce code.**

**Nous utilisons [GitHub issues](https://github.com/tensorflow/tensorflow/issues) 
pour le suivi des demandes et des Bugs. veuillez consulter
[TensorFlow Forum](https://discuss.tensorflow.org/) pour les questions 
discussions générales et veuillez adresser vous questions spécifiques à
[Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).**

Le projet TensorFlow s'efforce de respecter les meilleures pratiques généralement 
acceptées en matière de développement de logiciels libres.

##  Lignes directrices en matière de correctifs

Suivez les étapes suivantes pour corriger une version spécifique de TensorFlow, 
par exemple pour appliquer des correctifs à des Bugs ou à des vulnérabilités 
de sécurité :

*   Clonez le repo TensorFlow et passez à la branche correspondante pour la version
    version de TensorFlow souhaitée, par exemple, la branche `r2.8` pour la version 2.8.
*   Appliquer (c'est-à-dire sélectionner) les modifications souhaitées et résoudre tout conflit
    de code.
*   Exécuter les tests TensorFlow et s'assurer qu'ils réussissent.
*   [Build](https://www.tensorflow.org/install/source) le paquet pip TensorFlow à 
    partir des ressources

## État de la construction en continu

Vous pouvez trouver d'autres plateformes et configurations supportées par la communauté dans le
[TensorFlow SIG Build community builds table](https://github.com/tensorflow/build#community-supported-tensorflow-builds).

### Official Builds

Build Type                    | État                                                                                                                                                                             | Artifacts                                                                                                                                                                         
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

## Resources

*   [TensorFlow.org](https://www.tensorflow.org)
*   [Tutoriels TensorFlow](https://www.tensorflow.org/tutorials/)
*   [Modèles officiels TensorFlow](https://github.com/tensorflow/models/tree/master/official)
*   [Exemples TensorFlow](https://github.com/tensorflow/examples)
*   [Ateliers de programmation TensorFlow](https://codelabs.developers.google.com/?cat=TensorFlow)
*   [Blog TensorFlow](https://blog.tensorflow.org)
*   [Apprendre le ML avec TensorFlow](https://www.tensorflow.org/resources/learn-ml)
*   [TensorFlow Twitter](https://twitter.com/tensorflow)
*   [TensorFlow YouTube](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)
*   [Feuille de route pour l'optimisation du modèle TensorFlow](https://www.tensorflow.org/model_optimization/guide/roadmap)
*   [TensorFlow White Papers](https://www.tensorflow.org/about/bib)
*   [Boite à outils de visualisation TensorBoard](https://github.com/tensorflow/tensorboard)
*   [Recherche de code TensorFlow](https://cs.opensource.google/tensorflow/tensorflow)

En savoir plus sur le
[TensorFlow community](https://www.tensorflow.org/community) et comment y
[contribuer](https://www.tensorflow.org/community/contribute).

## Cours

*   [Deep Learning avec Tensorflow de Edx](https://www.edx.org/course/deep-learning-with-tensorflow)
*   [Certificat DeepLearning.AI TensorFlow Developpeur Professionnel de Coursera](https://www.coursera.org/specializations/tensorflow-in-practice)
*   [TensorFlow: Données et deploiement de Coursera](https://www.coursera.org/specializations/tensorflow-data-and-deployment)
*   [Demarer avec TensorFlow 2 de Coursera](https://www.coursera.org/learn/getting-started-with-tensor-flow2)
*   [TensorFlow: Techniques avancées de Coursera](https://www.coursera.org/specializations/tensorflow-advanced-techniques)
*   [TensorFlow 2 poue le Deep Learning Specialization de Coursera](https://www.coursera.org/specializations/tensorflow2-deeplearning)
*   [Introduction au TensorFlow pour A.I, M.L, and D.L de Coursera](https://www.coursera.org/learn/introduction-tensorflow)
*   [Machine Learning avec TensorFlow sur GCP de Coursera](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp)
*   [Introdution au TensorFlow por le Deep Learning d'Udacity](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
*   [Introduction au TensorFlow Lite d'Udacity](https://www.udacity.com/course/intro-to-tensorflow-lite--ud190)

## Licence

[Apache License 2.0](LICENSE)
