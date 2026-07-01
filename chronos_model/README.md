# Chronos 2
The Chronos-2 model used is based on Amazon's Chronos-2 (model id ``amazon/chronos-2``), accessed via
the ``chronos-forecasting`` package (``pip install "chronos-forecasting>=2.0"``)

See the Github and Hugging Face pages for more information at
https://github.com/amazon-science/chronos-forecasting
https://huggingface.co/amazon/chronos-2

## Model weights

This project uses Chronos-2 (amazon/chronos-2, ~120M params). The weights
are not committed to this repository — they are large and are downloaded once,
then cached locally.

Unlike the older Chronos / Chronos-Bolt models on the Hugging Face Hub,
Chronos-2 is distributed via S3 + CloudFront and downloaded with boto3.

You have two options:

**Option A — let the library handle it (recommended).**
Nothing to do. On first run, Chronos2Pipeline.from_pretrained("amazon/chronos-2")
downloads and caches the weights automatically. Requires outbound network access
on first run.

**Option B — keep a local copy (offline / pinned reproducibility)**.
Download once and save into the directory referenced by
pathmanager.model_utils:

```{python}
from chronos import Chronos2Pipeline
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
pipeline.save_pretrained("[path matching pathmanager.model_utils]")
```

The model loader checks for model.safetensors in that directory and uses the
local copy if present; otherwise it falls back to downloading from the hub.