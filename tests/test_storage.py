"""Storage helpers."""
from __future__ import annotations

from app.services import storage


def test_save_and_list_frames(tmp_data):
    job_id = storage.new_job_id()
    data = [(f"f{i}.jpg", b"\xff\xd8\xff") for i in range(3)]
    paths = storage.save_frames(job_id, data)
    assert len(paths) == 3
    listed = storage.frame_paths(job_id)
    assert len(listed) == 3
