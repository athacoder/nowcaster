"""One polite HTTP client for every source: identified, slow, no retries on a block."""

import ssl
import time

import httpx

USER_AGENT = "food-nowcaster/0.1 (student research project; low-rate)"


class PoliteClient:
    def __init__(
        self, min_interval: float = 2.0, timeout: float = 60.0, legacy_tls: bool = False
    ) -> None:
        context = ssl.create_default_context()
        if legacy_tls:
            # api.mospi.gov.in does not support secure renegotiation (RFC 5746), which
            # OpenSSL 3 refuses by default. Certificates are still verified.
            context.options |= ssl.OP_LEGACY_SERVER_CONNECT
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
            verify=context,
        )
        self._min_interval = min_interval
        self._last = 0.0

    def get(self, url: str, params: dict[str, str | int] | None = None) -> httpx.Response:
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            response = self._client.get(url, params=params)
        finally:
            self._last = time.monotonic()
        # 403 and 429 mean "stop", so they are raised, never retried.
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()
