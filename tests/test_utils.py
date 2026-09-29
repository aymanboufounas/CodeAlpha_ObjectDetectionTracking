from src.utils import parse_source


def test_webcam_source_is_integer():
    assert parse_source("0") == 0
    assert parse_source(" 2 ") == 2


def test_video_source_stays_string():
    assert parse_source("samples/demo.mp4") == "samples/demo.mp4"
