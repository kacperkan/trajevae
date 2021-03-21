import json
import typing as t

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests


def decode_image(byte_data: t.List[float]) -> np.ndarray:
    byte_data = np.asarray(byte_data, dtype=np.uint8)[..., np.newaxis]
    img = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
    return img


def render_zip():
    with open("examples/scene.zip", "rb") as f:
        result = requests.post(
            "http://localhost:8000/render_zip", files={"zip": f}
        )
    data = json.loads(result.content)
    an_img = decode_image(data)

    plt.figure()
    plt.imshow(an_img)
    plt.show()


def render_clear_xml():
    with open("examples/hello.xml") as f:
        result = requests.post("http://localhost:8000/render", data=f.read())
    data = json.loads(result.content)
    an_img = decode_image(data)

    plt.figure()
    plt.imshow(an_img)
    plt.show()


if __name__ == "__main__":
    render_clear_xml()
    render_zip()
