from tensorflow.python.ops.data_flow_ops import QueueBase, _as_type_list, _as_shape_list, _as_name_list

class StreamQueue(QueueBase):
  def __init__(self, stream_id, stream_columns, dtypes=None, capacity=100, 
               shapes=None, names=None, name="stream_queue"):
    if not dtypes:
      dtypes = [tf.int64, tf.float32]

    if not shapes:
      shapes = [[1], [1]]

    dtypes = _as_type_list(dtypes)
    shapes = _as_shape_list(shapes, dtypes)
    names = _as_name_list(names, dtypes)
    queue_ref = _op_def_lib.apply_op("Stream", stream_id=stream_id, 
                                     stream_columns=stream_columns, capacity=capacity,
                                     component_types=dtypes, shapes=shapes,
                                     name=name, container=None, shared_name=None)
    super(StreamQueue, self).__init__(dtypes, shapes, 
                                      names, queue_ref)

