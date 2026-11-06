# Contributing

Use English for documentation, identifiers, comments, and interface text. Keep model adapters separate from conversation scheduling. Include cancellation, bounded-buffer, and failure behavior tests for changes to streaming code.

Report hardware, model IDs, versions, cold versus warm runs, utterance lengths, p50/p95, and audio quality observations with performance changes. Do not label synthetic fixture results as human speech accuracy measurements. Never commit credentials, private audio, downloaded weights, or another project's source without its required license notices.

Run `uv run pytest` and `uv run ruff check .` before submitting a pull request. Hardware inference is not exercised by the lightweight CI suite.
