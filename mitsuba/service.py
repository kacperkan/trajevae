import os
import shutil
import subprocess
import warnings
from pathlib import Path

import cv2
from flask import Flask, jsonify, make_response, request

app = Flask(__name__)

TEMP_IMAGE = "/tmp/temp.exr"
MAPPED_TEMP_IMAGE = "/tmp/temp.png"
TEMP_SCENE = "/tmp/scene.xml"


def encode_image(path_to_xml: str) -> bytes:
    print("Rendering ...")
    return_code = subprocess.call(["mitsuba", "-o", TEMP_IMAGE, path_to_xml])
    print("Return code: {}".format(return_code))

    # some flags of rendering causes that png is produced directly
    if os.path.exists(TEMP_IMAGE):
        print("Tone mapping ...")
        return_code = subprocess.call(
            [
                "mtsutil",
                "tonemap",
                "-a",
                "-f",
                "png",
                "-o",
                MAPPED_TEMP_IMAGE,
                TEMP_IMAGE,
            ]
        )
        print("Return code: {}".format(return_code))
        os.remove(TEMP_IMAGE)
    an_img = cv2.imread(MAPPED_TEMP_IMAGE, cv2.IMREAD_UNCHANGED)[..., ::-1]

    os.remove(MAPPED_TEMP_IMAGE)

    encoded = cv2.imencode(".png", an_img)[1].squeeze().tolist()
    return encoded


@app.route("/render", methods=["GET", "POST"])
def render():
    xml_data = request.data
    xml_data = xml_data.decode()
    try:
        with open(TEMP_SCENE, "w") as f:
            f.write(xml_data)
        encoded = encode_image(TEMP_SCENE)

        return jsonify(encoded)
    except Exception as e:
        print(e)
        return make_response(jsonify(error=str(e)))
    finally:
        if os.path.exists(TEMP_SCENE):
            os.remove(TEMP_SCENE)
    return make_response()


@app.route("/render_zip", methods=["GET", "POST"])
def render_zip():
    zip_file = request.files["zip"]

    temporary_file_name = "/tmp/render_data.zip"
    unpack_directory = "/tmp/render_data/"

    zip_file.save(temporary_file_name)
    try:
        print("Unpacking ...")
        os.mkdir(unpack_directory)
        subprocess.call(["unzip", temporary_file_name, "-d", unpack_directory])

        xml_files = list(Path(unpack_directory).rglob("*.xml"))

        if len(xml_files) > 1:
            warnings.warn(
                f"Found XML {len(xml_files)} files in total, taking the first found"
            )

        xml_file = xml_files[0]

        encoded = encode_image(xml_file)
        shutil.rmtree(unpack_directory)
        os.remove(temporary_file_name)

        return jsonify(encoded)
    except Exception as e:
        print(e)
        return make_response(jsonify(error=str(e)))
    finally:
        if os.path.exists(unpack_directory):
            shutil.rmtree(unpack_directory)
        if os.path.exists(temporary_file_name):
            os.remove(temporary_file_name)
    return make_response()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000")
