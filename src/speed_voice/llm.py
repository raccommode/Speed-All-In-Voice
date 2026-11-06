import json
import re

import httpx


class PhraseBuffer:
    """Flush short complete clauses early, then cap text waiting for punctuation."""

    def __init__(self):
        self.text = ""
        self.first = True

    def push(self, text, final=False):
        self.text += text
        chunks = []
        while self.text:
            boundary = re.search(r"[.!?。！？]\s|[,;:]\s", self.text)
            limit = 60 if self.first else 140
            cut = boundary.end() if boundary else 0
            if not cut and len(self.text) >= limit:
                cut = self.text.rfind(" ", 0, limit + 1)
                if cut <= 0:
                    cut = limit
            if not cut and final:
                cut = len(self.text)
            if not cut:
                break
            part, self.text = self.text[:cut].strip(), self.text[cut:]
            if part:
                chunks.append(part)
                self.first = False
        return chunks


class LLM:
    def __init__(self, settings):
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=5))

    async def close(self):
        await self.client.aclose()

    async def tokens(self, messages):
        s = self.settings
        async with self.client.stream(
            "POST",
            s.llm_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + s.llm_key},
            json={
                "model": s.llm_model,
                "messages": messages,
                "stream": True,
                "max_tokens": s.max_reply_tokens,
                "temperature": 0.3,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError("The language model returned an error")
                choices = event.get("choices", [])
                if choices:
                    text = choices[0].get("delta", {}).get("content")
                    if text:
                        yield text
