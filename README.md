# Fourth attempt at making Neural Particle Automata


Okay, so here are my thoughts:

We are not _just_ making stable NPA. We are making "biologically sound" cell-like particles for generalized morphogenesis.

IDEAS:
- Two explicit perpendicular polarities.
- Isotropic.
- Potential formulation (because it is cool?)
- 3D
- "GraphConv" or "NNConv" or "  " for best interpretability

Lessons learned:
- Explicit message oritentations
- Explicit degree 
- Use **all** the tricks from Alexander Mordvintsev for training!
- Zero-initialization of last layer can be important

MAYBE:
- Hidden states as GRN?
- Cell division

## Static Dashboard Viewer (GitHub Pages)

This repository includes a minimal static dashboard picker for exported Panel HTML files.

### Expected layout

- `docs/index.html` - picker UI
- `docs/manifest.json` - generated run index
- `docs/runs/<run_id>/display.html` - exported dashboards

### Add a new run

1. Copy your exported dashboard into `docs/runs/<run_id>/display.html`.
2. Regenerate the manifest:

```bash
python scripts/generate_dashboard_manifest.py
```

3. Commit `docs/runs/...` and `docs/manifest.json`.

### Optional metadata

You can place `docs/runs/<run_id>/meta.json` to enrich list entries.

Example:

```json
{
	"title": "saving_test",
	"description": "Quick smoke-test training run",
	"tags": ["smoke", "oval"],
	"model_path": "notebooks/runs/saving_test/model.pt",
	"config_path": "notebooks/runs/saving_test/model.pt_config.pt"
}
```
