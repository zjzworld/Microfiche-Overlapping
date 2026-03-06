# Microfiche Overlap Extractor (Windows EXE)

## What it does

1. Scan PDF pages with a selected vision-capable LLM (pure LLM, no geometry rule in final decision).
2. Generate overlap CSV: source directory, file name, page number, confidence.
3. Export overlap pages as single-page PDFs:
- `O_<original_stem>_P<page>.pdf`
4. Export non-overlap pages into cleaned files:
- `E_<original_filename>.pdf`
5. Replace overlap pages using candidate pages from source or optional replacement directory:
- `R_<original_filename>.pdf`
6. Export blurry unreadable pages as single-page PDFs:
- `B_<original_stem>_P<page>.pdf`
7. Training and memory:
- Import corrections CSV to create hard overrides.
- Add global notes/rules used in future prompt context.

## Build on Windows

Run in the project folder (or GitHub Workflows):

```bat
build_exe.bat
```

Output EXE:

```text
dist\MicroficheOverlapExtractor.exe
```

## GUI workflow

1. Select model profile (built-in `Kimi-K2.5` included).
2. Fill API key.
3. Optional: add/edit model profiles.
4. Select source directory.
5. Select actions to run.
6. Optional: set replacement directory and custom prompt.
7. Optional: enable `Live output while scanning` for immediate CSV / O_ / E_ / B_ outputs.
8. Optional: enable `Fast mode (lower DPI)` for faster processing.
9. Click `Run`.

## Training CSV format

CSV headers (flexible; any equivalent names are accepted):

- `file_name` or `file` or `file_path`
- `page`
- `is_overlap` or `label` (`true/false`, `yes/no`, `1/0`, or `blurry`)
- `note` (optional)

Example:

```csv
file_name,page,is_overlap,note
SANDO, ROSEMARY M - SANDULIAK, ALBERT D.pdf,3,true,Known ghost overlap
SANDO, ROSEMARY M - SANDULIAK, ALBERT D.pdf,4,false,Single clean card
```

## Notes

- Some third-party gateways are unstable for image inputs. Use the `Test Model` button first.
- Memory and model configs are saved under user app data.
- GUI model names are display aliases (e.g. `GPT-5.3`, `Claude-Opus-4.6`); the app internally maps them to real model IDs.
- In live mode, outputs are produced during scan instead of waiting for the entire batch to finish.
- `Blurry` means the page is unreadable enough that student name and grades cannot be recognized at all. If any name or grades are visible, it should not be marked blurry.
