
## Annotation data formats

There are several data annotation formats allowed for TensorBox. The most simple is json-file format which presents the
data structure well.

### Currnet json-format

TensorBox expects the list of objects each of which describes one annotations for one image.

```json
[image]
  
image{
  "image_path": string,
  "rects": [rect]
}
  
rect{
  "x1": int,
  "y1": int,
  "x2": int,
  "y2": int
}
```
Each annotation is an object with two properties: `image_path` (string) and `rects` (list). The second property
describes all bounding boxes which present on the current image. The format of bounding box description consists of
four integer properties which mean the main diagonal of the rectangle `(x1, y1) - (x2,y2)`. TensorBox reading procedure
expects that `x1<x2` and `y1<y2`. Example:

```json
[
  {
    "image_path": "images/1/abc.jpg",
    "rects":
      [
        {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        {"x1": 200, "y1": 150, "x2": 220, "y2": 300}
      ]
  },
  {
    "image_path": "images/2/klm.jpg",
    "rects":
      [
        {"x1": 200, "y1": 0, "x2": 300, "y2": 100}
      ]
  },
]
```

### Future extension

In the comming future the following extension will take place:
```json
{
"images": [image],
"classes": [string]
}
  
image{
  "id": int,
  "image_path": string,
  "rects": [rect]
}
  
rect{
  "classID": int,
  "x1": int,
  "y1": int,
  "x2": int,
  "y2": int
}
```
This extension allows to point class of object which is surrounded by each box. The `classID` value is the index in
`classes` collection. This extension is necessary step towards to the multiclass TensorBox model which we hope will be
implemented later.
