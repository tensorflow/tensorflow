[1mdiff --git a/tensorflow/python/tf2.py b/tensorflow/python/tf2.py[m
[1mindex bce72186790..8eae643b019 100644[m
[1m--- a/tensorflow/python/tf2.py[m
[1m+++ b/tensorflow/python/tf2.py[m
[36m@@ -20,19 +20,21 @@[m [mthe TensorFlow library. For that see tf.compat instead.[m
 [m
 from tensorflow.python.platform import _pywrap_tf2[m
 from tensorflow.python.util.tf_export import tf_export[m
[32m+[m[32mimport tensorflow as tf[m
 [m
 [m
 def enable():[m
   # Enables v2 behaviors.[m
[31m-  _pywrap_tf2.enable(True)[m
[32m+[m[32m  tf.compat.v1.enable_v2_behavior()[m
[32m+[m
 [m
 [m
 def disable():[m
   # Disables v2 behaviors.[m
[31m-  _pywrap_tf2.enable(False)[m
[32m+[m[32m  tf.compat.v1.disable_v2_behavior()[m
 [m
 [m
 @tf_export("__internal__.tf2.enabled", v1=[])[m
 def enabled():[m
   # Returns True iff TensorFlow 2.0 behavior should be enabled.[m
[31m-  return _pywrap_tf2.is_enabled()[m
[32m+[m[32m  return tf.executing_eagerly()[m
