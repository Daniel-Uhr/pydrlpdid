# Contributing

Contributions that improve documentation, testing, diagnostics, and numerical
reliability are welcome.

## Development installation

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest -q
```

## Pull requests

Please keep pull requests focused and include tests for behavioral changes.
Changes to estimator definitions, supported samples, weighting, or inference
must document their econometric implications and must not be introduced as
silent changes to a released specification.
