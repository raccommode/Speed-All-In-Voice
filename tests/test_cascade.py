"""The experiment must never speak routing markers or a rejected draft."""

import runpy

import pytest

Cloud = runpy.run_path("benchmarks/openrouter_cascade.py")["Cloud"]


@pytest.mark.parametrize(
    "prefix,expected_models,expected_reply",
    [
        (["ANS", "WER\n", "Paris."], ["small"], "Paris."),
        (["DE", "LEGATE"], ["small", "large"], "Twenty-three."),
        (["Invalid decision\nDo not speak this."], ["small", "large"], "Twenty-three."),
    ],
)
async def test_cascade_routes_without_speaking_control_text(
    prefix, expected_models, expected_reply
):
    cloud = Cloud("test-key", "small", "large")
    cloud.mode = "routed"
    calls = []
    original = [
        {"role": "system", "content": "Voice assistant"},
        {"role": "user", "content": "Question"},
    ]

    async def request(model, messages):
        calls.append(model)
        if model == "large":
            assert messages == original
        for token in prefix if model == "small" else ["Twenty-three."]:
            yield token

    cloud.request = request
    try:
        text = "".join([token async for token in cloud.tokens(original)])
        assert text == expected_reply
        assert calls == expected_models
        assert cloud.route == expected_models[-1]
    finally:
        await cloud.close()


def test_openrouter_key_is_only_mapped_to_exact_service(monkeypatch):
    from speed_voice.config import Settings

    monkeypatch.delenv("SPEED_LLM_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setenv("SPEED_LLM_URL", "https://openrouter.ai/api/v1")
    assert Settings.from_env().llm_key == "test-secret"
    monkeypatch.setenv("SPEED_LLM_URL", "https://example.com/v1")
    assert Settings.from_env().llm_key == "ollama"
    monkeypatch.setenv("SPEED_LLM_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("SPEED_LLM_KEY", "explicit-test-key")
    assert Settings.from_env().llm_key == "explicit-test-key"
