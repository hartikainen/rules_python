import importlib.metadata


def test_py_spy_metadata_is_available():
    assert importlib.metadata.version("py-spy") == "0.4.1"
