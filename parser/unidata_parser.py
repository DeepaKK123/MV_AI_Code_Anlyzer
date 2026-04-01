"""
parser/unidata_parser.py
UniData BASIC static analyser.

Extracts structural metadata from .B source files:
  - Subroutine name
  - CALL targets
  - OPEN / CLOSE file handle tracking
  - READ / READU / MATREAD patterns
  - WRITE / WRITEU / MATWRITE patterns
  - LOOP statement line numbers (for EXIT guard review)
  - READU locked-read detection
  - Unclosed file handle detection (open_count > close_count)

Usage (standalone test):
    python -m parser.unidata_parser path/to/MY.SUBROUTINE.B
"""

import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class SubroutineInfo:
    name: str
    file_path: str
    calls: list = field(default_factory=list)
    opens: list = field(default_factory=list)
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    loops: list = field(default_factory=list)
    readu_files: list = field(default_factory=list)
    unclosed: bool = False


def parse_unidata_file(file_path: str) -> SubroutineInfo:
    """Parse a single UniData .B source file and return structured metadata."""
    path = Path(file_path)
    info = SubroutineInfo(name=path.stem, file_path=file_path)
    open_count = 0
    close_count = 0

    with open(file_path, "r", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            upper = line.upper().strip()

            # Skip comment lines (UniData BASIC uses * for comments)
            if upper.startswith("*") or upper.startswith("!"):
                continue

            # CALL statements
            m = re.search(r"\bCALL\s+([\w\.]+)", upper)
            if m:
                target = m.group(1)
                if target not in info.calls:
                    info.calls.append(target)

            # OPEN file handle
            m = re.search(r"\bOPEN\b.*\bTO\s+(\w+)", upper)
            if m:
                info.opens.append(m.group(1))
                open_count += 1

            # CLOSE file handle
            if re.search(r"\bCLOSE\b", upper):
                close_count += 1

            # READ variants (READ, READU, MATREAD, MATREADU)
            m = re.search(r"\b(MATREADU?|READU|READ)\s+\w+\s+FROM\s+(\w+)", upper)
            if m:
                file_var = m.group(2)
                info.reads.append(file_var)
                if "READU" in m.group(1):
                    info.readu_files.append(file_var)

            # WRITE variants (WRITE, WRITEU, MATWRITE, MATWRITEU)
            m = re.search(r"\b(MATWRITEU?|WRITEU|WRITE)\s+\w+\s+(?:ON|TO)\s+(\w+)", upper)
            if m:
                info.writes.append(m.group(2))

            # LOOP - flag line numbers for review
            if re.search(r"\bLOOP\b", upper):
                info.loops.append(lineno)

    info.unclosed = open_count > close_count
    return info


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python -m parser.unidata_parser <path/to/file.B>")
        sys.exit(1)
    result = parse_unidata_file(sys.argv[1])
    print(json.dumps(result.__dict__, indent=2))
