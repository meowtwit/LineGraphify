# LineGraphify

Python GUI app and experimental C++ processing core for contour-based graph-art formula generation.

## Python GUI

```bash
python3 main.py
```

## C++ Processing Core

The C++ core is a standalone CLI intended for large formula counts where the Python processing path becomes slow.

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
