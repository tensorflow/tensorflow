import sys
#import AnnoList_pb2
import AnnotationLib;

from ma_utils import is_number;

def loadPal(filename):
    _annolist = AnnoList_pb2.AnnoList();

    f = open(filename, "rb");
    _annolist.ParseFromString(f.read());
    f.close();

    return _annolist;

def savePal(filename, _annolist):
    f = open(filename, "wb");
    f.write(_annolist.SerializeToString());
    f.close();

def al2pal(annotations):
    _annolist = AnnoList_pb2.AnnoList();

    #assert(isinstance(annotations, AnnotationLib.AnnoList));

    # check type of attributes, add missing attributes
    for a in annotations:
        for r in a.rects:
            for k, v in r.at.iteritems():
                if not k in annotations.attribute_desc:
                    annotations.add_attribute(k, type(v));
                else:
                    assert(AnnotationLib.is_compatible_attr_type(annotations.attribute_desc[k].dtype, type(v)));

    # check attributes values
    for a in annotations:
        for r in a.rects:
            for k, v in r.at.iteritems():
                if k in annotations.attribute_val_to_str:
                    # don't allow undefined values
                    if not v in annotations.attribute_val_to_str[k]:
                        print "attribute: {}, undefined value: {}".format(k, v);
                        assert(False);

    # store attribute descriptions in pal structure
    for aname, adesc in annotations.attribute_desc.iteritems():
        _annolist.attribute_desc.extend([adesc]);

    for a in annotations:
        _a = _annolist.annotation.add();
        _a.imageName = a.imageName;

        for r in a.rects:
            _r = _a.rect.add();

            _r.x1 = r.x1;
            _r.y1 = r.y1;
            _r.x2 = r.x2;
            _r.y2 = r.y2;

            _r.score = float(r.score);

            if hasattr(r, 'id'):
                _r.id = r.id;

            if hasattr(r, 'track_id'):
                _r.track_id = r.track_id;

            if hasattr(r, 'at'):
                for k, v in r.at.items():
                    _at = _r.attribute.add();

                    _at.id = annotations.attribute_desc[k].id;

                    if annotations.attribute_desc[k].dtype == AnnotationLib.AnnoList.TYPE_INT32:
                        assert(AnnotationLib.is_compatible_attr_type(AnnotationLib.AnnoList.TYPE_INT32, type(v)));
                        _at.val = int(v);
                    elif annotations.attribute_desc[k].dtype == AnnotationLib.AnnoList.TYPE_FLOAT:
                        assert(AnnotationLib.is_compatible_attr_type(AnnotationLib.AnnoList.TYPE_FLOAT, type(v)));
                        _at.fval = float(v);
                    elif annotations.attribute_desc[k].dtype == AnnotationLib.AnnoList.TYPE_STRING:
                        assert(AnnotationLib.is_compatible_attr_type(AnnotationLib.AnnoList.TYPE_STRING, type(v)));
                        _at.strval = str(v);
                    else:
                        assert(false);

    return _annolist;

def pal2al(_annolist):
    #annotations = [];
    annotations = AnnotationLib.AnnoList();

    for adesc in _annolist.attribute_desc:
        annotations.attribute_desc[adesc.name] = adesc;
        print "attribute: ", adesc.name, adesc.id

        for valdesc in adesc.val_to_str:
            annotations.add_attribute_val(adesc.name, valdesc.s, valdesc.id);

    attribute_name_from_id = {adesc.id: aname for aname, adesc in annotations.attribute_desc.iteritems()}
    attribute_dtype_from_id = {adesc.id: adesc.dtype for aname, adesc in annotations.attribute_desc.iteritems()}

    for _a in _annolist.annotation:
        anno = AnnotationLib.Annotation()

        anno.imageName = _a.imageName;

        anno.rects = [];

        for _r in _a.rect:
            rect = AnnotationLib.AnnoRect()

            rect.x1 = _r.x1;
            rect.x2 = _r.x2;
            rect.y1 = _r.y1;
            rect.y2 = _r.y2;

            if _r.HasField("id"):
                rect.id = _r.id;

            if _r.HasField("track_id"):
                rect.track_id = _r.track_id;

            if _r.HasField("score"):
                rect.score = _r.score;

            for _at in _r.attribute:
                try:
                    cur_aname = attribute_name_from_id[_at.id];
                    cur_dtype = attribute_dtype_from_id[_at.id];
                except KeyError as e:
                    print "attribute: ", _at.id
                    print e
                    assert(False);

                if cur_dtype == AnnotationLib.AnnoList.TYPE_INT32:
                    rect.at[cur_aname] = _at.val;
                elif cur_dtype == AnnotationLib.AnnoList.TYPE_FLOAT:
                    rect.at[cur_aname] = _at.fval;
                elif cur_dtype == AnnotationLib.AnnoList.TYPE_STRING:
                    rect.at[cur_aname] = _at.strval;
                else:
                    assert(False);

            anno.rects.append(rect);

        annotations.append(anno);

    return annotations;
