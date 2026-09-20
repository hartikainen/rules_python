import argparse
from pathlib import Path

from python import runfiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=Path, required=True)
    args, _ = parser.parse_known_args()
    files = runfiles.Create()
    assert files is not None
    payload = files.Rlocation("_main/tests/uv/lock/testdata/toolchain_payload.txt")
    assert payload is not None
    args.output_file.write_bytes(Path(payload).read_bytes())


if __name__ == "__main__":
    main()
