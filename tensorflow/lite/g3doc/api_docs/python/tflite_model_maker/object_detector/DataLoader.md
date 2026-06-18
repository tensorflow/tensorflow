page_type: reference
description: DataLoader for object detector.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.object_detector.DataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="from_cache"/>
<meta itemprop="property" content="from_csv"/>
<meta itemprop="property" content="from_pascal_voc"/>
<meta itemprop="property" content="gen_dataset"/>
<meta itemprop="property" content="split"/>
</div>

# tflite_model_maker.object_detector.DataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/object_detector_dataloader.py#L102-L367">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



DataLoader for object detector.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.object_detector.DataLoader(
    tfrecord_file_patten, size, label_map, annotations_json_file=None
)
</code></pre>




<h3>Used in the notebooks</h3>
<table class="vertical-rules">
  <thead>
    <tr>
      <th>Used in the tutorials</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
  <ul>
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/object_detection">Object Detection with TensorFlow Lite Model Maker</a></li>
  </ul>
</td>
    </tr>
  </tbody>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tfrecord_file_patten`<a id="tfrecord_file_patten"></a>
</td>
<td>
Glob for tfrecord files. e.g. "/tmp/coco*.tfrecord".
</td>
</tr><tr>
<td>
`size`<a id="size"></a>
</td>
<td>
The size of the dataset.
</td>
</tr><tr>
<td>
`label_map`<a id="label_map"></a>
</td>
<td>
Variable shows mapping label integers ids to string label
names. 0 is the reserved key for `background` and doesn't need to be
included in label_map. Label names can't be duplicated. Supported
formats are:

1. Dict, map label integers ids to string label names, such as {1:
  'person', 2: 'notperson'}. 2. List, a list of label names such as
    ['person', 'notperson'] which is
   the same as setting label_map={1: 'person', 2: 'notperson'}.
3. String, name for certain dataset. Accepted values are: 'coco', 'voc'
  and 'waymo'. 4. String, yaml filename that stores label_map.
</td>
</tr><tr>
<td>
`annotations_json_file`<a id="annotations_json_file"></a>
</td>
<td>
JSON with COCO data format containing golden
bounding boxes. Used for validation. If None, use the ground truth from
the dataloader. Refer to
https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
  for the description of COCO data format.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`size`<a id="size"></a>
</td>
<td>
Returns the size of the dataset.

Note that this function may return None becuase the exact size of the
dataset isn't a necessary parameter to create an instance of this class,
and tf.data.Dataset donesn't support a function to get the length directly
since it's lazy-loaded and may be infinite.
In most cases, however, when an instance of this class is created by helper
functions like 'from_folder', the size of the dataset will be preprocessed,
and this function can return an int representing the size of the dataset.
</td>
</tr>
</table>



## Methods

<h3 id="from_cache"><code>from_cache</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/object_detector_dataloader.py#L307-L336">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_cache(
    cache_prefix
)
</code></pre>

Loads the data from cache.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`cache_prefix`
</td>
<td>
The cache prefix including the cache directory and the cache
prefix filename, e.g: '/tmp/cache/train'.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
ObjectDetectorDataLoader object.
</td>
</tr>

</table>



<h3 id="from_csv"><code>from_csv</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/object_detector_dataloader.py#L224-L305">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_csv(
    filename: str,
    images_dir: Optional[str] = None,
    delimiter: str = &#x27;,&#x27;,
    quotechar: str = &#x27;\&#x27;&quot;,
    num_shards: int = 10,
    max_num_images: Optional[int] = None,
    cache_dir: Optional[str] = None,
    cache_prefix_filename: Optional[str] = None
) -> List[Optional[DetectorDataLoader]]
</code></pre>

Loads the data from the csv file.

The csv format is shown in
<a href="https://cloud.google.com/vision/automl/object-detection/docs/csv-format">https://cloud.google.com/vision/automl/object-detection/docs/csv-format</a> We
supports bounding box with 2 vertices for now. We support the files in the
local machine as well.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`filename`
</td>
<td>
Name of the csv file.
</td>
</tr><tr>
<td>
`images_dir`
</td>
<td>
Path to directory that store raw images. If None, the image
path in the csv file is the path to Google Cloud Storage or the absolute
path in the local machine.
</td>
</tr><tr>
<td>
`delimiter`
</td>
<td>
Character used to separate fields.
</td>
</tr><tr>
<td>
`quotechar`
</td>
<td>
Character used to quote fields containing special characters.
</td>
</tr><tr>
<td>
`num_shards`
</td>
<td>
Number of shards for output file.
</td>
</tr><tr>
<td>
`max_num_images`
</td>
<td>
Max number of imags to process.
</td>
</tr><tr>
<td>
`cache_dir`
</td>
<td>
The cache directory to save TFRecord, metadata and json file.
When cache_dir is None, a temporary folder will be created and will not
be removed automatically after training which makes it can be used
later.
</td>
</tr><tr>
<td>
`cache_prefix_filename`
</td>
<td>
The cache prefix filename. If None, will
automatically generate it based on `filename`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
train_data, validation_data, test_data which are ObjectDetectorDataLoader
objects. Can be None if without such data.
</td>
</tr>

</table>



<h3 id="from_pascal_voc"><code>from_pascal_voc</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/object_detector_dataloader.py#L137-L222">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_pascal_voc(
    images_dir: str,
    annotations_dir: str,
    label_map: Union[List[str], Dict[int, str], str],
    annotation_filenames: Optional[Collection[str]] = None,
    ignore_difficult_instances: bool = False,
    num_shards: int = 100,
    max_num_images: Optional[int] = None,
    cache_dir: Optional[str] = None,
    cache_prefix_filename: Optional[str] = None
) -> DetectorDataLoader
</code></pre>

Loads from dataset with PASCAL VOC format.

Refer to
<a href="https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5">https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5</a>
for the description of PASCAL VOC data format.

LabelImg Tool (<a href="https://github.com/tzutalin/labelImg">https://github.com/tzutalin/labelImg</a>) can annotate the image
and save annotations as XML files in PASCAL VOC data format.

Annotations are in the folder: `annotations_dir`.
Raw images are in the foloder: `images_dir`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`images_dir`
</td>
<td>
Path to directory that store raw images.
</td>
</tr><tr>
<td>
`annotations_dir`
</td>
<td>
Path to the annotations directory.
</td>
</tr><tr>
<td>
`label_map`
</td>
<td>
Variable shows mapping label integers ids to string label
names. 0 is the reserved key for `background`. Label names can't be
duplicated. Supported format: 1. Dict, map label integers ids to string
  label names, e.g.
   {1: 'person', 2: 'notperson'}. 2. List, a list of label names. e.g.
     ['person', 'notperson'] which is
   the same as setting label_map={1: 'person', 2: 'notperson'}.

3. String, name for certain dataset. Accepted values are: 'coco', 'voc'
  and 'waymo'. 4. String, yaml filename that stores label_map.
</td>
</tr><tr>
<td>
`annotation_filenames`
</td>
<td>
Collection of annotation filenames (strings) to be
loaded. For instance, if there're 3 annotation files [0.xml, 1.xml,
2.xml] in `annotations_dir`, setting annotation_filenames=['0', '1']
makes this method only load [0.xml, 1.xml].
</td>
</tr><tr>
<td>
`ignore_difficult_instances`
</td>
<td>
Whether to ignore difficult instances.
`difficult` can be set inside `object` item in the annotation xml file.
</td>
</tr><tr>
<td>
`num_shards`
</td>
<td>
Number of shards for output file.
</td>
</tr><tr>
<td>
`max_num_images`
</td>
<td>
Max number of imags to process.
</td>
</tr><tr>
<td>
`cache_dir`
</td>
<td>
The cache directory to save TFRecord, metadata and json file.
When cache_dir is not set, a temporary folder will be created and will
not be removed automatically after training which makes it can be used
later.
</td>
</tr><tr>
<td>
`cache_prefix_filename`
</td>
<td>
The cache prefix filename. If not set, will
automatically generate it based on `image_dir`, `annotations_dir` and
`annotation_filenames`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
ObjectDetectorDataLoader object.
</td>
</tr>

</table>



<h3 id="gen_dataset"><code>gen_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/object_detector_dataloader.py#L338-L362">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gen_dataset(
    model_spec, batch_size=None, is_training=False, use_fake_data=False
)
</code></pre>

Generate a batched tf.data.Dataset for training/evaluation.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_spec`
</td>
<td>
Specification for the model.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
A integer, the returned dataset will be batched by this size.
</td>
</tr><tr>
<td>
`is_training`
</td>
<td>
A boolean, when True, the returned dataset will be optionally
shuffled and repeated as an endless dataset.
</td>
</tr><tr>
<td>
`use_fake_data`
</td>
<td>
Use fake input.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A TF dataset ready to be consumed by Keras model.
</td>
</tr>

</table>



<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/object_detector_dataloader.py#L364-L367">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    fraction
)
</code></pre>

This function isn't implemented for the object detection task.


<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/dataloader.py#L126-L130">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>
