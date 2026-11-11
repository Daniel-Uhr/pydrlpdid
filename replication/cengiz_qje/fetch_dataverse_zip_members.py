"""List or selectively fetch members of a large remote ZIP via HTTP ranges.

This helper avoids downloading the complete 2 GB ``data1.zip`` from the
official Cengiz et al. (2019) replication package when only a few documented
Stata files are required.  It parses a locally saved tail of the ZIP central
directory and verifies the CRC-32 of every extracted member.
"""

from __future__ import annotations

import argparse
import binascii
import struct
import urllib.request
import zlib
from pathlib import Path


CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"
CENTRAL_FIXED = struct.Struct("<6H3I5H2I")
LOCAL_FIXED = struct.Struct("<5H3I2H")


def parse_central_directory(tail_path: Path) -> dict[str, dict[str, int]]:
    raw = tail_path.read_bytes()
    entries: dict[str, dict[str, int]] = {}
    pos = 0
    while True:
        pos = raw.find(CENTRAL_SIGNATURE, pos)
        if pos < 0:
            break
        start = pos + len(CENTRAL_SIGNATURE)
        if start + CENTRAL_FIXED.size > len(raw):
            break
        fields = CENTRAL_FIXED.unpack_from(raw, start)
        (
            _version_made,
            _version_needed,
            flags,
            method,
            _mtime,
            _mdate,
            crc32,
            compressed_size,
            uncompressed_size,
            name_len,
            extra_len,
            comment_len,
            _disk_start,
            _internal_attr,
            _external_attr,
            local_offset,
        ) = fields
        end = start + CENTRAL_FIXED.size + name_len + extra_len + comment_len
        if end > len(raw):
            pos += 4
            continue
        name_bytes = raw[
            start + CENTRAL_FIXED.size : start + CENTRAL_FIXED.size + name_len
        ]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        entries[name] = {
            "flags": flags,
            "method": method,
            "crc32": crc32,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        }
        pos = end
    return entries


def http_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "pydrlpdid-replication/0.8 (selective Dataverse download)",
        },
    )
    with urllib.request.urlopen(request) as response:
        payload = response.read()
        if len(payload) != end - start + 1:
            raise RuntimeError(
                f"Server returned {len(payload)} bytes for range {start}-{end}."
            )
        return payload


def fetch_member(url: str, name: str, meta: dict[str, int], output: Path) -> None:
    offset = meta["local_offset"]
    header = http_range(url, offset, offset + 29)
    if header[:4] != LOCAL_SIGNATURE:
        raise RuntimeError(f"Invalid local ZIP header for {name!r}.")
    fields = LOCAL_FIXED.unpack_from(header, 4)
    name_len, extra_len = fields[-2:]
    data_start = offset + 30 + name_len + extra_len
    compressed_size = meta["compressed_size"]
    compressed = http_range(url, data_start, data_start + compressed_size - 1)

    if meta["method"] == 0:
        content = compressed
    elif meta["method"] == 8:
        content = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported ZIP compression method {meta['method']}.")

    if len(content) != meta["uncompressed_size"]:
        raise RuntimeError(f"Uncompressed size mismatch for {name!r}.")
    if binascii.crc32(content) & 0xFFFFFFFF != meta["crc32"]:
        raise RuntimeError(f"CRC-32 mismatch for {name!r}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--list", dest="pattern")
    parser.add_argument("--url")
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    entries = parse_central_directory(args.tail)
    if args.pattern is not None:
        pattern = args.pattern.casefold()
        for name, meta in sorted(entries.items()):
            if pattern in name.casefold():
                print(
                    f"{name}\tcompressed={meta['compressed_size']}\t"
                    f"uncompressed={meta['uncompressed_size']}"
                )

    if args.member:
        if not args.url or args.output_dir is None:
            parser.error("--url and --output-dir are required with --member")
        for name in args.member:
            if name not in entries:
                raise KeyError(f"ZIP member not found: {name}")
            output = args.output_dir / name
            print(f"Fetching {name} -> {output}")
            fetch_member(args.url, name, entries[name], output)
            print(f"Verified {name}")


if __name__ == "__main__":
    main()
