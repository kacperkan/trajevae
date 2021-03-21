# Mitsuba service

Since [mitsuba]() library is a bit out-of-date, I created a containerized flask service that has an endpoint to the mitsuba binary. There's a single endpoint which allows to send an XML file and a rendered image is returned.

## Installation

1. `docker build -t mitsuba-flask:1.0 .`
2. `docker run -d -p 8000:8000 mitsuba-flask:1.0`

After the installation, the following entrypoints  will be available:

1. `http://localhost:8000/render` - it allows to render an image from an XML format of string.
2. `http://localhost:8000/render_zip` - allows to render complex scenes where files are packed into the zip file. It assumes that a single XML file exists there. 

## Processing
These endpoints return encoded, jsonified images. To decode the image, use (for XML file):
```python
result = requests.post("http://localhost:8000/render", data=xml_string)
data = json.loads(result.content)
encoding = np.asarray(data, dtype=np.uint8)[..., np.newaxis]
img = cv2.imdecode(encoding, cv2.IMREAD_COLOR)
```
and for zip file:
```python
with open(<your_zip_file>, "rb") as f:
    result = requests.post(
        "http://localhost:8000/render_zip", files={"zip": f}
    )
data = json.loads(result.content)
encoding = np.asarray(data, dtype=np.uint8)[..., np.newaxis]
img = cv2.imdecode(encoding, cv2.IMREAD_COLOR)
```

## Examples

Usage examples are provided in `test.py`. It uses pip dependencies from `requirements.txt`.

## Acknowledgements

My fork is influenced by: [amyspark](https://github.com/amyspark/mitsuba/) 
I used their fixes to run v0.6 of the mitsuba 

The Dockerfile is influenced by: [ninjaben](https://hub.docker.com/r/ninjaben/mitsuba-rgb/dockerfile) 
I copied almost whole Dockerfile and adapted it to run on the Ubuntu 16.04 container.
