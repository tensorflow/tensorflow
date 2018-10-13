import numpy as np
import random
import json
import os
import cv2
import itertools
from scipy.misc import imread, imresize
import tensorflow as tf

from data_utils import (annotation_jitter, annotation_to_h5)
from utils.annolist import AnnotationLib as al
from rect import Rect
from utils import tf_concat

def rescale_boxes(current_shape, anno, target_height, target_width, test=False):
    x_scale = target_width / float(current_shape[1])
    y_scale = target_height / float(current_shape[0])
    for r in anno.rects:
        if r.x1 >= r.x2:
            if test:
                r.x1, r.x2 = r.x2, r.x1
            else:   
                assert r.x1 < r.x2
        r.x1 *= x_scale
        r.x2 *= x_scale
        if r.y1 >= r.y2:
            if test:
                r.y1, r.y2 = r.y2, r.y1
            else:   
                assert r.y1 < r.y2
        r.y1 *= y_scale
        r.y2 *= y_scale
    return anno

def load_idl_tf(idlfile, H, jitter):
    """Take the idlfile and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    annolist = al.parse(idlfile)
    annos = []
    for anno in annolist:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
        annos.append(anno)
    random.seed(0)
    if H['data']['truncate_data']:
        annos = annos[:10]
    for epoch in itertools.count():
        random.shuffle(annos)
        for anno in annos:
            try:
                if 'grayscale' in H and 'grayscale_prob' in H:
                    I = imread(anno.imageName, mode = 'RGB' if random.random() < H['grayscale_prob'] else 'L')
                    if len(I.shape) < 3:
                        I = cv2.cvtColor(I, cv2.COLOR_GRAY2RGB)
                else:
                    if len(I.shape) < 3:
                        continue
                    I = imread(anno.imageName, mode = 'RGB')
                if I.shape[0] != H["image_height"] or I.shape[1] != H["image_width"]:
                    if epoch == 0:
                        anno = rescale_boxes(I.shape, anno, H["image_height"], H["image_width"])
                    I = imresize(I, (H["image_height"], H["image_width"]), interp='cubic')
                if jitter:
                    jitter_scale_min=0.9
                    jitter_scale_max=1.1
                    jitter_offset=16
                    I, anno = annotation_jitter(I,
                                                anno, target_width=H["image_width"],
                                                target_height=H["image_height"],
                                                jitter_scale_min=jitter_scale_min,
                                                jitter_scale_max=jitter_scale_max,
                                                jitter_offset=jitter_offset)

                boxes, flags = annotation_to_h5(H,
                                                anno,
                                                H["grid_width"],
                                                H["grid_height"],
                                                H["rnn_len"])

                yield {"image": I, "boxes": boxes, "flags": flags}
            except Exception as exc:
                print(exc)

def make_sparse(n, d):
    v = np.zeros((d,), dtype=np.float32)
    v[n] = 1.
    return v

def load_data_gen(H, phase, jitter):
    grid_size = H['grid_width'] * H['grid_height']

    data = load_idl_tf(H["data"]['%s_idl' % phase], H, jitter={'train': jitter, 'test': False}[phase])

    for d in data:
        output = {}

        rnn_len = H["rnn_len"]
        flags = d['flags'][0, :, 0, 0:rnn_len, 0]
        boxes = np.transpose(d['boxes'][0, :, :, 0:rnn_len, 0], (0, 2, 1))
        assert(flags.shape == (grid_size, rnn_len))
        assert(boxes.shape == (grid_size, rnn_len, 4))

        output['image'] = d['image']
        output['confs'] = np.array([[make_sparse(int(detection), d=H['num_classes']) for detection in cell] for cell in flags])
        output['boxes'] = boxes
        output['flags'] = flags

        yield output

def add_rectangles(H, orig_image, confidences, boxes, use_stitching=False, rnn_len=1, min_conf=0.1, show_removed=True, tau=0.25, show_suppressed=True):
    image = np.copy(orig_image[0])
    num_cells = H["grid_height"] * H["grid_width"]
    boxes_r = np.reshape(boxes, (-1,
                                 H["grid_height"],
                                 H["grid_width"],
                                 rnn_len,
                                 4))
    confidences_r = np.reshape(confidences, (-1,
                                             H["grid_height"],
                                             H["grid_width"],
                                             rnn_len,
                                             H['num_classes']))
    cell_pix_size = H['region_size']
    all_rects = [[[] for _ in range(H["grid_width"])] for _ in range(H["grid_height"])]
    for n in range(rnn_len):
        for y in range(H["grid_height"]):
            for x in range(H["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                w = bbox[2]
                h = bbox[3]
                conf = np.max(confidences_r[0, y, x, n, 1:])
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))

    all_rects_r = [r for row in all_rects for cell in row for r in cell]
    if use_stitching:
        from stitch_wrapper import stitch_rects
        acc_rects = stitch_rects(all_rects, tau)
    else:
        acc_rects = all_rects_r


    if show_suppressed:
        pairs = [(all_rects_r, (255, 0, 0))]
    else:
        pairs = []
    pairs.append((acc_rects, (0, 255, 0)))
    for rect_set, color in pairs:
        for rect in rect_set:
            if rect.confidence > min_conf:
                cv2.rectangle(image,
                    (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                    (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                    color,
                    1)

    cv2.putText(image,str(len(filter(lambda rect:rect.confidence > min_conf, acc_rects))), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    rects = []
    for rect in acc_rects:
        if rect.confidence > min_conf:
            r = al.AnnoRect()
            r.x1 = rect.cx - rect.width/2.
            r.x2 = rect.cx + rect.width/2.
            r.y1 = rect.cy - rect.height/2.
            r.y2 = rect.cy + rect.height/2.
            r.score = rect.confidence
            rects.append(r)

    return image, rects

def to_x1y1x2y2(box):
    w = tf.maximum(box[:, 2:3], 1)
    h = tf.maximum(box[:, 3:4], 1)
    x1 = box[:, 0:1] - w / 2
    x2 = box[:, 0:1] + w / 2
    y1 = box[:, 1:2] - h / 2
    y2 = box[:, 1:2] + h / 2
    return tf_concat(1, [x1, y1, x2, y2])

def intersection(box1, box2):
    x1_max = tf.maximum(box1[:, 0], box2[:, 0])
    y1_max = tf.maximum(box1[:, 1], box2[:, 1])
    x2_min = tf.minimum(box1[:, 2], box2[:, 2])
    y2_min = tf.minimum(box1[:, 3], box2[:, 3])

    x_diff = tf.maximum(x2_min - x1_max, 0)
    y_diff = tf.maximum(y2_min - y1_max, 0)

    return x_diff * y_diff

def area(box):
    x_diff = tf.maximum(box[:, 2] - box[:, 0], 0)
    y_diff = tf.maximum(box[:, 3] - box[:, 1], 0)
    return x_diff * y_diff

def union(box1, box2):
    return area(box1) + area(box2) - intersection(box1, box2)

def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)

def to_idx(vec, w_shape):
    '''
    vec = (idn, idh, idw)
    w_shape = [n, h, w, c]
    '''
    return vec[:, 2] + w_shape[2] * (vec[:, 1] + w_shape[1] * vec[:, 0])

def interp(w, i, channel_dim):
    '''
    Input:
        w: A 4D block tensor of shape (n, h, w, c)
        i: A list of 3-tuples [(x_1, y_1, z_1), (x_2, y_2, z_2), ...],
            each having type (int, float, float)

        The 4D block represents a batch of 3D image feature volumes with c channels.
        The input i is a list of points  to index into w via interpolation. Direct
        indexing is not possible due to y_1 and z_1 being float values.
    Output:
        A list of the values: [
            w[x_1, y_1, z_1, :]
            w[x_2, y_2, z_2, :]
            ...
            w[x_k, y_k, z_k, :]
        ]
        of the same length == len(i)
    '''
    w_as_vector = tf.reshape(w, [-1, channel_dim]) # gather expects w to be 1-d
    upper_l = tf.to_int32(tf_concat(1, [i[:, 0:1], tf.floor(i[:, 1:2]), tf.floor(i[:, 2:3])]))
    upper_r = tf.to_int32(tf_concat(1, [i[:, 0:1], tf.floor(i[:, 1:2]), tf.ceil(i[:, 2:3])]))
    lower_l = tf.to_int32(tf_concat(1, [i[:, 0:1], tf.ceil(i[:, 1:2]), tf.floor(i[:, 2:3])]))
    lower_r = tf.to_int32(tf_concat(1, [i[:, 0:1], tf.ceil(i[:, 1:2]), tf.ceil(i[:, 2:3])]))

    upper_l_idx = to_idx(upper_l, tf.shape(w))
    upper_r_idx = to_idx(upper_r, tf.shape(w))
    lower_l_idx = to_idx(lower_l, tf.shape(w))
    lower_r_idx = to_idx(lower_r, tf.shape(w))

    upper_l_value = tf.gather(w_as_vector, upper_l_idx)
    upper_r_value = tf.gather(w_as_vector, upper_r_idx)
    lower_l_value = tf.gather(w_as_vector, lower_l_idx)
    lower_r_value = tf.gather(w_as_vector, lower_r_idx)

    alpha_lr = tf.expand_dims(i[:, 2] - tf.floor(i[:, 2]), 1)
    alpha_ud = tf.expand_dims(i[:, 1] - tf.floor(i[:, 1]), 1)

    upper_value = (1 - alpha_lr) * upper_l_value + (alpha_lr) * upper_r_value
    lower_value = (1 - alpha_lr) * lower_l_value + (alpha_lr) * lower_r_value
    value = (1 - alpha_ud) * upper_value + (alpha_ud) * lower_value
    return value

def bilinear_select(H, pred_boxes, early_feat, early_feat_channels, w_offset, h_offset):
    '''
    Function used for rezooming high level feature maps. Uses bilinear interpolation
    to select all channels at index (x, y) for a high level feature map, where x and y are floats.
    '''
    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']

    fine_stride = 8. # pixels per 60x80 grid cell in 480x640 image
    coarse_stride = H['region_size'] # pixels per 15x20 grid cell in 480x640 image
    batch_ids = []
    x_offsets = []
    y_offsets = []
    for n in range(H['batch_size']):
        for i in range(H['grid_height']):
            for j in range(H['grid_width']):
                for k in range(H['rnn_len']):
                    batch_ids.append([n])
                    x_offsets.append([coarse_stride / 2. + coarse_stride * j])
                    y_offsets.append([coarse_stride / 2. + coarse_stride * i])

    batch_ids = tf.constant(batch_ids)
    x_offsets = tf.constant(x_offsets)
    y_offsets = tf.constant(y_offsets)

    pred_boxes_r = tf.reshape(pred_boxes, [outer_size * H['rnn_len'], 4])
    scale_factor = coarse_stride / fine_stride # scale difference between 15x20 and 60x80 features

    pred_x_center = (pred_boxes_r[:, 0:1] + w_offset * pred_boxes_r[:, 2:3] + x_offsets) / fine_stride
    pred_x_center_clip = tf.clip_by_value(pred_x_center,
                                     0,
                                     scale_factor * H['grid_width'] - 1)
    pred_y_center = (pred_boxes_r[:, 1:2] + h_offset * pred_boxes_r[:, 3:4] + y_offsets) / fine_stride
    pred_y_center_clip = tf.clip_by_value(pred_y_center,
                                          0,
                                          scale_factor * H['grid_height'] - 1)

    interp_indices = tf_concat(1, [tf.to_float(batch_ids), pred_y_center_clip, pred_x_center_clip])
    return interp_indices
