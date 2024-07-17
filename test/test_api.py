import unittest
import pytest


def test_infer(client, image_path):
    with open(image_path, "rb") as image_file:
        files = {"file": (image_path, image_file, "image/png")}
        response = client.post("/api/infer/", files=files)
        assert response.status_code == 200