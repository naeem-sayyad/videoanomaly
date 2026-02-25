from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass
class VLMResult:
    text: str
    status_code: Optional[int]
    force_unknown: bool
    error: str


class VLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        mode: str,
        custom_endpoint: str,
        timeout_sec: int,
        logger,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mode = mode.lower().strip() or "openai"
        self.custom_endpoint = custom_endpoint if custom_endpoint.startswith("/") else f"/{custom_endpoint}"
        self.timeout_sec = timeout_sec
        self.logger = logger

    def infer(self, prompt: str, image_paths: Optional[List[str]] = None) -> VLMResult:
        images = image_paths or []

        if self.mode == "openai":
            response = self._post_openai(prompt, images)
        else:
            response = self._post_custom(prompt, images)

        if images and response.status_code is not None and 400 <= response.status_code < 500:
            self.logger.warning(
                "VLM server rejected image payload with %s; falling back to unknown label.",
                response.status_code,
            )
            self._probe_text_only(prompt)
            return VLMResult(text="", status_code=response.status_code, force_unknown=True, error="image_input_unsupported")

        return response

    def _probe_text_only(self, prompt: str) -> None:
        try:
            if self.mode == "openai":
                self._post_openai(prompt, [])
            else:
                self._post_custom(prompt, [])
        except Exception:
            return

    def _post_openai(self, prompt: str, image_paths: List[str]) -> VLMResult:
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_openai_payload(prompt, image_paths)

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_sec)
            if not resp.ok:
                return VLMResult(text="", status_code=resp.status_code, force_unknown=False, error=resp.text[:500])

            data = resp.json()
            text = self._extract_openai_text(data)
            return VLMResult(text=text, status_code=resp.status_code, force_unknown=False, error="")
        except requests.RequestException as exc:
            return VLMResult(text="", status_code=None, force_unknown=False, error=str(exc))
        except ValueError as exc:
            return VLMResult(text="", status_code=None, force_unknown=False, error=f"Invalid JSON response: {exc}")

    def _post_custom(self, prompt: str, image_paths: List[str]) -> VLMResult:
        url = f"{self.base_url}{self.custom_endpoint}"
        payload = self._build_custom_payload(prompt, image_paths)

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_sec)
            if not resp.ok:
                return VLMResult(text="", status_code=resp.status_code, force_unknown=False, error=resp.text[:500])

            data = resp.json()
            text = self._extract_custom_text(data)
            return VLMResult(text=text, status_code=resp.status_code, force_unknown=False, error="")
        except requests.RequestException as exc:
            return VLMResult(text="", status_code=None, force_unknown=False, error=str(exc))
        except ValueError as exc:
            return VLMResult(text="", status_code=None, force_unknown=False, error=f"Invalid JSON response: {exc}")

    def _build_openai_payload(self, prompt: str, image_paths: List[str]) -> Dict[str, Any]:
        if image_paths:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image_path in image_paths:
                b64 = self._encode_image_base64(image_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
        else:
            content = prompt

        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 400,
        }
        if self.model:
            payload["model"] = self.model
        return payload

    def _build_custom_payload(self, prompt: str, image_paths: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": 0.0,
            "n_predict": 400,
        }
        if self.model:
            payload["model"] = self.model

        if image_paths:
            payload["images"] = [self._encode_image_base64(p) for p in image_paths]

        return payload

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def _extract_openai_text(response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _extract_custom_text(response_json: Dict[str, Any]) -> str:
        if "content" in response_json:
            return str(response_json.get("content", ""))
        if "response" in response_json:
            return str(response_json.get("response", ""))
        if "text" in response_json:
            return str(response_json.get("text", ""))

        choices = response_json.get("choices", [])
        if choices and isinstance(choices, list):
            first = choices[0]
            if isinstance(first, dict):
                return str(first.get("text", ""))

        return ""
