import json

from src.llm_client import LLMClient


def test_llm_client_generic_endpoint(monkeypatch):
    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers

        class _Resp:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json_module.dumps({"result": "ok"})
                            }
                        }
                    ]
                }

        return _Resp()

    json_module = json
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1/chat")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setattr("src.llm_client.requests.post", fake_post)

    client = LLMClient(model="generic-model", temperature=0.0)
    response = client.chat_json("system", "user")

    assert calls["url"] == "https://example.com/v1/chat"
    assert calls["headers"]["Authorization"] == "Bearer secret"
    assert calls["json"]["model"] == "generic-model"
    assert response["result"] == "ok"
