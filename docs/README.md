# Documentation index

| Doc | Purpose |
|-----|---------|
| [../README.md](../README.md) | Project overview, inputs, run commands, outputs |
| [README.pdf](README.pdf) | Downloadable PDF of the project README |
| [RUNNING.md](RUNNING.md) | Detailed recipes, compute, troubleshooting |
| [OUTPUTS.md](OUTPUTS.md) | Artifact formats |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | Caveats that affect interpretation |
| [../data/README.md](../data/README.md) | CAMELS download + directory layout |

`data/readme.txt` is the upstream CAMELS attributes citation / history file
shipped beside the download instructions — not a project run guide.

Regenerate the PDF after README edits:

```bash
python scripts/build_readme_pdf.py
# writes docs/README.pdf
```
