# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""API defining dataset features (image, text, scalar,...).

See [the guide](https://www.tensorflow.org/datasets/features).

"""

from tensorflow.data.experimental.core.features.audio_feature import Audio
from tensorflow.data.experimental.core.features.bounding_boxes import BBox
from tensorflow.data.experimental.core.features.bounding_boxes import BBoxFeature
from tensorflow.data.experimental.core.features.class_label_feature import ClassLabel
from tensorflow.data.experimental.core.features.feature import FeatureConnector
from tensorflow.data.experimental.core.features.feature import Tensor
from tensorflow.data.experimental.core.features.feature import TensorInfo
from tensorflow.data.experimental.core.features.features_dict import FeaturesDict
from tensorflow.data.experimental.core.features.image_feature import Image
from tensorflow.data.experimental.core.features.sequence_feature import Sequence
from tensorflow.data.experimental.core.features.text_feature import Text
from tensorflow.data.experimental.core.features.translation_feature import Translation
from tensorflow.data.experimental.core.features.translation_feature import TranslationVariableLanguages
from tensorflow.data.experimental.core.features.video_feature import Video

__all__ = [
    "Audio",
    "BBox",
    "BBoxFeature",
    "ClassLabel",
    "FeatureConnector",
    "FeaturesDict",
    "Tensor",
    "TensorInfo",
    "Sequence",
    "Image",
    "Text",
    "Video",
]
