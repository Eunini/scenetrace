from scenetrace.store import Store


def test_opening_metadata_does_not_invalidate_a_running_service_job(tmp_path):
    store = Store(tmp_path)
    store.execute(
        "INSERT INTO jobs (id,owner,kind,state,created,updated) VALUES (?,?,?,?,?,?)",
        ("job", "owner", "index", "running", 1, 1),
    )
    other = Store(tmp_path)
    assert other.one("SELECT state FROM jobs WHERE id='job'")["state"] == "running"
    other.recover()
    assert other.one("SELECT state FROM jobs WHERE id='job'")["state"] == "failed"
