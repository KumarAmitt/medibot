from __future__ import annotations

import os


def configure_model_downloads() -> None:
    """Point TLS at certifi so Hugging Face downloads work on typical Windows installs."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    try:
        import certifi

        ca_file = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca_file)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_file)
        os.environ.setdefault("CURL_CA_BUNDLE", ca_file)
    except Exception:
        pass
