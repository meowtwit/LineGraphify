# LineGraphify

Python GUI app and experimental C++ processing core for contour-based graph-art formula generation.

## Python GUI

```bash
python3 main.py
```

## Hybrid Python + C++ Flow

The default workflow is now hybrid:

- Python GUI handles image loading, OpenCV edge extraction, contour extraction, previews, and PNG output.
- `build/formula_generator` handles large formula-file generation from contour points without depending on OpenCV C++.

Build the C++ generator:

```bash
cmake -S . -B build
cmake --build build -j
```

Then run the GUI:

```bash
python3 main.py
```

In the GUI, enable `C++で大量数式生成` for large formula counts. Python will make a smaller preview, and the C++ generator will write the selected formula data format when you press `数式データ保存`.

## C++ OpenCV Processing Core

`linegraphify_core` is an experimental full C++ OpenCV path. It is only built when a C++ OpenCV package is available.

### macOS prerequisites

Homebrew OpenCV needs the Xcode Command Line Tools, not only Xcode.app:

```bash
xcode-select --install
brew install opencv
```

### Build

```bash
cmake -S . -B build
cmake --build build -j
```

### Example

```bash
./build/linegraphify_core \
  --input image.png \
  --output-png out.png \
  --output-json formulas.json \
  --max-formulas 100000 \
  --fill-zones 1 \
  --samples-per-segment 6
```

For `100000` to `1000000` formulas, JSON/TXT output size and write time can dominate the runtime. If you only need a preview, omit `--output-json` and `--output-txt`.
