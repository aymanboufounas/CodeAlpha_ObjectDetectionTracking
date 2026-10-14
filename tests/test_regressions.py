import pytest

from src.utils import ensure_output_path, parse_source
from src.tracker import ObjectTracker
from src.detector import Detection


def test_source_rejects_blank_input():
    with pytest.raises(ValueError):
        parse_source("   ")


def test_camera_source_and_filename():
    assert parse_source(" 12 ") == 12
    assert parse_source("clip01.mp4") == "clip01.mp4"


def test_output_must_be_mp4(tmp_path):
    with pytest.raises(ValueError):
        ensure_output_path(str(tmp_path / "wrong.avi"))
    result = ensure_output_path(str(tmp_path / "nested" / "out.mp4"))
    assert result.parent.is_dir()


def test_deepsort_rejects_invalid_boxes():
    good = Detection((1, 2, 10, 20), 0.9, 0, "person")
    zero_width = Detection((5, 2, 5, 20), 0.9, 0, "person")
    reversed_box = Detection((10, 2, 1, 20), 0.9, 0, "person")
    invalid_conf = Detection((1, 2, 10, 20), float("nan"), 0, "person")
    assert ObjectTracker._to_deepsort(
        [good, zero_width, reversed_box, invalid_conf]
    ) == [([1, 2, 9, 18], 0.9, "person")]
