import os
import cv2
import re
import sys
import argparse
import numpy as np
import copy
import json
import annolist.AnnotationLib as al
from xml.etree import ElementTree
from scipy.misc import imread

def annotation_to_h5(H, a, cell_width, cell_height, max_len):
    region_size = H['region_size']
    assert H['region_size'] == H['image_height'] / H['grid_height']
    assert H['region_size'] == H['image_width'] / H['grid_width']
    cell_regions = get_cell_grid(cell_width, cell_height, region_size)

    cells_per_image = len(cell_regions)

    box_list = [[] for idx in range(cells_per_image)]
            
    for cidx, c in enumerate(cell_regions):
        box_list[cidx] = [r for r in a.rects if all(r.intersection(c))]

    boxes = np.zeros((1, cells_per_image, 4, max_len, 1), dtype = np.float)
    box_flags = np.zeros((1, cells_per_image, 1, max_len, 1), dtype = np.float)

    for cidx in xrange(cells_per_image):
        #assert(cur_num_boxes <= max_len)

        cell_ox = 0.5 * (cell_regions[cidx].x1 + cell_regions[cidx].x2)
        cell_oy = 0.5 * (cell_regions[cidx].y1 + cell_regions[cidx].y2)

        unsorted_boxes = []
        for bidx in xrange(min(len(box_list[cidx]), max_len)):

            # relative box position with respect to cell
            ox = 0.5 * (box_list[cidx][bidx].x1 + box_list[cidx][bidx].x2) - cell_ox
            oy = 0.5 * (box_list[cidx][bidx].y1 + box_list[cidx][bidx].y2) - cell_oy

            width = abs(box_list[cidx][bidx].x2 - box_list[cidx][bidx].x1)
            height= abs(box_list[cidx][bidx].y2 - box_list[cidx][bidx].y1)
            
            if (abs(ox) < H['focus_size'] * region_size and abs(oy) < H['focus_size'] * region_size and
                    width < H['biggest_box_px'] and height < H['biggest_box_px']):
                unsorted_boxes.append(np.array([ox, oy, width, height], dtype=np.float))

        for bidx, box in enumerate(sorted(unsorted_boxes, key=lambda x: x[0]**2 + x[1]**2)):
            boxes[0, cidx, :, bidx, 0] = box
            box_flags[0, cidx, 0, bidx, 0] = max(box_list[cidx][bidx].silhouetteID, 1)

    return boxes, box_flags

def get_cell_grid(cell_width, cell_height, region_size):

    cell_regions = []
    for iy in xrange(cell_height):
        for ix in xrange(cell_width):
            cidx = iy * cell_width + ix
            ox = (ix + 0.5) * region_size
            oy = (iy + 0.5) * region_size

            r = al.AnnoRect(ox - 0.5 * region_size, oy - 0.5 * region_size,
                            ox + 0.5 * region_size, oy + 0.5 * region_size)
            r.track_id = cidx

            cell_regions.append(r)


    return cell_regions

def annotation_jitter(I, a_in, min_box_width=20, jitter_scale_min=0.9, jitter_scale_max=1.1, jitter_offset=16, target_width=640, target_height=480):
    a = copy.deepcopy(a_in)

    # MA: sanity check
    new_rects = []
    for i in range(len(a.rects)):
        r = a.rects[i]
        try:
            assert(r.x1 < r.x2 and r.y1 < r.y2)
            new_rects.append(r)
        except:
            print('bad rectangle')
    a.rects = new_rects


    if a.rects:
        cur_min_box_width = min([r.width() for r in a.rects])
    else:
        cur_min_box_width = min_box_width / jitter_scale_min

    # don't downscale below min_box_width 
    jitter_scale_min = max(jitter_scale_min, float(min_box_width) / cur_min_box_width)

    # it's always ok to upscale 
    jitter_scale_min = min(jitter_scale_min, 1.0)

    jitter_scale_max = jitter_scale_max

    jitter_scale = np.random.uniform(jitter_scale_min, jitter_scale_max)

    jitter_flip = np.random.random_integers(0, 1)

    if jitter_flip == 1:
        I = np.fliplr(I)

        for r in a:
            r.x1 = I.shape[1] - r.x1
            r.x2 = I.shape[1] - r.x2
            r.x1, r.x2 = r.x2, r.x1

            for p in r.point:
                p.x = I.shape[1] - p.x

    I1 = cv2.resize(I, None, fx=jitter_scale, fy=jitter_scale, interpolation = cv2.INTER_CUBIC)

    jitter_offset_x = np.random.random_integers(-jitter_offset, jitter_offset)
    jitter_offset_y = np.random.random_integers(-jitter_offset, jitter_offset)



    rescaled_width = I1.shape[1]
    rescaled_height = I1.shape[0]

    px = round(0.5*(target_width)) - round(0.5*(rescaled_width)) + jitter_offset_x
    py = round(0.5*(target_height)) - round(0.5*(rescaled_height)) + jitter_offset_y

    I2 = np.zeros((target_height, target_width, 3), dtype=I1.dtype)

    x1 = max(0, px)
    y1 = max(0, py)
    x2 = min(rescaled_width, target_width - x1)
    y2 = min(rescaled_height, target_height - y1)

    I2[0:(y2 - y1), 0:(x2 - x1), :] = I1[y1:y2, x1:x2, :]

    ox1 = round(0.5*rescaled_width) + jitter_offset_x
    oy1 = round(0.5*rescaled_height) + jitter_offset_y

    ox2 = round(0.5*target_width)
    oy2 = round(0.5*target_height)

    for r in a:
        r.x1 = round(jitter_scale*r.x1 - x1)
        r.x2 = round(jitter_scale*r.x2 - x1)

        r.y1 = round(jitter_scale*r.y1 - y1)
        r.y2 = round(jitter_scale*r.y2 - y1)

        if r.x1 < 0:
            r.x1 = 0

        if r.y1 < 0:
            r.y1 = 0

        if r.x2 >= I2.shape[1]:
            r.x2 = I2.shape[1] - 1

        if r.y2 >= I2.shape[0]:
            r.y2 = I2.shape[0] - 1

        for p in r.point:
            p.x = round(jitter_scale*p.x - x1)
            p.y = round(jitter_scale*p.y - y1)

        # MA: make sure all points are inside the image
        r.point = [p for p in r.point if p.x >=0 and p.y >=0 and p.x < I2.shape[1] and p.y < I2.shape[0]]

    new_rects = []
    for r in a.rects:
        if r.x1 <= r.x2 and r.y1 <= r.y2:
            new_rects.append(r)
        else:
            pass

    a.rects = new_rects

    return I2, a

def convert_sloth(filename):
    with open(filename) as f:
      annos = json.load(f)
    new_annos = [
        {
          "image_path" : anno["filename"],
          "rects" : [
            {
              'x1' : rect["x"],
              'x2' : rect["x"] + rect["width"],
              'y1' : rect["y"],
              'y2' : rect["y"] + rect["height"],
            } for rect in anno["annotations"]
          ]
        } for anno in annos
    ]
    with open("{}/{}".format(os.path.dirname(filename), "annotations.json"), 'w') as f:
        json.dump(new_annos, f)

def convert_to_sloth(filename):
    with open(filename) as f:
      annos = json.load(f)
    new_annos = [
        {
          "filename" : anno["image_path"],
          "annotations" : [
            {
              'x' : rect["x1"],
              'y' : rect["y1"],
              'width' : rect["x2"] - rect["x1"],
              'height' : rect["y2"] - rect["y1"],
            } for rect in anno["rects"]
          ]
        } for anno in annos
    ]
    with open("{}/{}".format(os.path.dirname(filename), "annotations_sloth.json"), 'w') as f:
        json.dump(new_annos, f)

def convert_pets2009(filename, version, dirname):
    root = ElementTree.parse(filename).getroot()
    res = [
        {
            "image_path" : "{}/frame_{:04d}.jpg".format(dirname, int(frame.attrib['number'])),
            "rects" : [
                {
                    'x1' : float(obj[0].attrib['xc']) - float(obj[0].attrib['w']) / 2,
                    'x2' : float(obj[0].attrib['xc']) + float(obj[0].attrib['w']) / 2,
                    'y1' : float(obj[0].attrib['yc']) - float(obj[0].attrib['h']) / 2,
                    'y2' : float(obj[0].attrib['yc']) + float(obj[0].attrib['h']) / 2,
                } for obj in frame[0].findall('object')
            ] 
        } for frame in root.findall('frame') 
    ]
    with open("data/annotation_PETS_{}.json".format(version), "w") as f:
        json.dump(res, f)


def convert_tud_campus(filename, dirname):
    root = ElementTree.parse(filename).getroot()
    res = [
        {
            "image_path" : "{}/DaSide0811-seq6-{:03d}.png".format(dirname, int(frame.attrib['number'])),
            "rects" : [
                {
                    'x1' : float(obj[0].attrib['xc']) - float(obj[0].attrib['w']) / 2,
                    'x2' : float(obj[0].attrib['xc']) + float(obj[0].attrib['w']) / 2,
                    'y1' : float(obj[0].attrib['yc']) - float(obj[0].attrib['h']) / 2,
                    'y2' : float(obj[0].attrib['yc']) + float(obj[0].attrib['h']) / 2,
                } for obj in frame[0].findall('object')
            ] 
        } for frame in root.findall('frame') 
    ]
    with open("data/annotation_TUD_CAMPUS.json", "w") as f:
        json.dump(res, f)


def convert_tud_crossing(filename, dirname):
    root = ElementTree.parse(filename).getroot()
    res = [
        {
            "image_path" : "{}/DaSide0811-seq7-{:03d}.png".format(dirname, int(frame.attrib['number'])),
            "rects" : [
                {
                    'x1' : float(obj[0].attrib['xc']) - float(obj[0].attrib['w']) / 2,
                    'x2' : float(obj[0].attrib['xc']) + float(obj[0].attrib['w']) / 2,
                    'y1' : float(obj[0].attrib['yc']) - float(obj[0].attrib['h']) / 2,
                    'y2' : float(obj[0].attrib['yc']) + float(obj[0].attrib['h']) / 2,
                } for obj in frame[0].findall('object')
            ] 
        } for frame in root.findall('frame') 
    ]
    with open("data/annotation_TUD_CROSSING.json", "w") as f:
        json.dump(res, f)

def convert_kitty(filename, version, dirname):
    res = []
    with open(filename) as f:
        for line in f:
            frame_nmb, r, tp, z, c, v, x1, y1, x2, y2, a, b, c, d, e, f, g = line.split()
            frame_nmb = int(frame_nmb)
            while frame_nmb >= len(res):
                res.append({
                    "image_path": "{}/{:06d}.png".format(dirname, len(res)),
                    "rects" : []
                })
            if tp == "Pedestrian":
                res[frame_nmb]["rects"].append({
                    'x1' : float(x1),
                    'x2' : float(x2),
                    'y1' : float(y1),
                    'y2' : float(y2)
                })
    with open("data/annotation_KITTY_{}.json".format(version), 'w') as f:
        json.dump(res, f)

def convert_pets2017(filename, version, dirname, last):
    res = []
    with open(filename) as f:
        for line in f:
            frame_nmb, pd_nmb, x1, y1, w, h, a, b, c, d = line.split(",")
            frame_nmb = int(frame_nmb)
            while frame_nmb > len(res):
                res.append({
                    "image_path": "{}/{:06d}.jpg".format(dirname, len(res) + 1),
                    "rects" : []
                })
            res[frame_nmb - 1]["rects"].append({
                'x1' : float(x1),
                'x2' : float(x1) + float(w),
                'y1' : float(y1),
                'y2' : float(y1) + float(h)
            })
    while(last > len(res)):
        res.append({
            "image_path": "{}/{:06d}.jpg".format(dirname, len(res) + 1),
            "rects" : []
        })
    with open("data/annotation_PETS2017_{}.json".format(version), 'w') as f:
        json.dump(res, f)

def convert_berkley_mat(filename, datadir
    ):
    from scipy.io import loadmat
    identity_ids, photoset_ids, owner_ids, photo_ids, head_boxes, train_idx, test_idx, val_idx, leftover_idx, test_split = loadmat(filename)['data'][0][0]
    res = {}
    for i in range(len(head_boxes)):
        image_name = "{}_{}".format(int(photoset_ids[i][0][0]), int(photo_ids[i][0][0]))
        bbox = head_boxes[i]
        if i + 1 in train_idx:
            prefix = "train"
        elif i + 1 in test_idx:
            prefix = "test"
        elif i + 1 in val_idx:
            prefix = "val"
        else:
            prefix = "leftover"
        image_path = "{}/{}.jpg".format(prefix, image_name)
        if image_path not in res:
            res[image_path] = []
        rect = {
            "x1" : float(bbox[0]),
            "x2" : float(bbox[0] + bbox[2]),
            "y1" : float(bbox[1]),
            "y2" : float(bbox[1] + bbox[3]),
        }
        if rect not in res[image_path]:
            res[image_path].append(rect)
    res2 = []
    for key, rects in res.iteritems():
        res2.append(
            {
                "image_path": key,
                "rects" : rects
            }
        )
    with open("{}/annos.json".format(datadir), 'w') as f:
        json.dump(res2, f)


def convert_berkley(textfile, annos, datadir):
    with open(annos) as f:
        annos = json.load(f)
    annos2 = {}
    for anno in annos:
        annos2[anno["image_path"]] = anno["rects"]
    prefixes = ['leftover', 'train', 'val', 'test']
    for line in open(textfile):
        photoset_id, photo_id, xmin, ymin, width, height, dentity_id, subset_id = map(int, line.split())
        image_path = "{}/{}_{}.jpg".format(prefixes[subset_id], photoset_id, photo_id)
        if image_path not in annos2:
            annos2[image_path] = []
        rect = {
            "x1" : float(xmin),
            "x2" : float(xmin + width),
            "y1" : float(ymin),
            "y2" : float(ymin + height),
        }
        if rect not in annos2[image_path]:
            annos2[image_path].append(rect)
    res = []
    for key, rects in annos2.iteritems():
        res.append(
            {
                "image_path": "{}/{}".format(datadir, key),
                "rects" : rects
            }
        )
    with open("{}/annos2.json".format(datadir), 'w') as f:
        json.dump(res, f)


def convert_hollywood(phase, datadir):
    annos = []
    for image_id in open("{}/Splits/{}.txt".format(datadir, phase)):
        image_id = image_id.rstrip('\n')
        image_path = "{}/JPEGImages/{}.jpeg".format(datadir, image_id)
        xml_path = "{}/Annotations/{}.xml".format(datadir, image_id)
        root = ElementTree.parse(xml_path).getroot()
        annos.append({
            "image_path" : image_path,
            "rects" : [{
                    'x1' : float(obj[1][0].text),
                    'y1' : float(obj[1][1].text),
                    'x2' : float(obj[1][2].text),
                    'y2' : float(obj[1][3].text)
                } for obj in root.findall('object') if obj.find("bndbox")] 
        })
    with open("{}/{}.json".format(datadir, phase), 'w') as f:
        json.dump(annos, f)


def merge_annotations(output_name, *files):
    res = []
    for json_anno in files:
        with open(json_anno) as f:
            anno = json.load(f)
            res += anno
    import random
    random.shuffle(res)
    with open(output_name, 'w') as f:
        json.dump(res, f)