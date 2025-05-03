from PIL import Image
import base64
import io


def save_image(file, path):
    img = Image.open(file)
    img.save(path)


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
