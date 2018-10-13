import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
from train import build_forward
import argparse
import json
import os

def create_frozen_graph(args, H):
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        freeze_graph(args.input, '', False, args.weights, "add,Reshape_2", "save/restore_all",
            "save/Const:0", args.output, False, '')

def create_graph(args, H):
    tf.reset_default_graph()
    H["grid_width"] = H["image_width"] / H["region_size"]
    H["grid_height"] = H["image_height"] / H["region_size"]
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    tf.add_to_collection('placeholders', x_in)
    tf.add_to_collection('vars', pred_boxes)
    tf.add_to_collection('vars', pred_confidences)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        metafile_path = args.input if args.frozen else args.output
        metafile_path = metafile_path.split('.pb')[0]
        saver.save(sess, metafile_path)
        tf.train.write_graph(sess.graph.as_graph_def(), '', args.input if args.frozen else args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--hypes', required=True)
    parser.add_argument('--weights', required=False)
    parser.add_argument('--frozen', type=bool, default=False)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--input', default='')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with open(args.hypes, 'r') as f:
        H = json.load(f)
    if args.frozen:
        if not os.path.exists(args.input):
            create_graph(args, H)
        create_frozen_graph(args, H)
    else:
        create_graph(args, H)