"""Streamlit client for the local RAG API."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen
from typing import Any

import streamlit as st


def _answer(api_url: str, question: str) -> dict[str, Any]:
    request = Request(
        f"{api_url.rstrip('/')}/answer",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urlopen(request) as response:  # noqa: S310 - URL is configured by the local user.
        for line in response:
            if line.startswith(b"data: "):
                result = json.loads(line.removeprefix(b"data: "))
                if not _valid_result(result):
                    raise ValueError("RAG service returned an invalid result event.")
                return result
    raise ValueError("RAG service did not return a result event.")


def _valid_result(result: Any) -> bool:
    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
        return False
    if result["status"] == "answered":
        if not isinstance(result.get("answer"), str):
            return False
    elif not isinstance(result.get("message"), str):
        return False
    sources = result.get("sources")
    rewrite = result.get("rewrite")
    return (
        isinstance(rewrite, dict)
        and rewrite.get("status") in {"rewritten", "fallback", "not_run"}
        and rewrite.get("route") in {"text", "visual", "table"}
        and isinstance(sources, list)
        and all(
        isinstance(source, dict)
        and isinstance(source.get("source_id"), str)
        and isinstance(source.get("document_id"), str)
        and isinstance(source.get("page_numbers"), list)
        and all(isinstance(page, int) for page in source["page_numbers"])
        and isinstance(source.get("type"), str)
        and isinstance(source.get("excerpt"), str)
        for source in sources
        )
    )


def main() -> None:
    st.title("Fitness document Q&A")
    api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
    question = st.text_area("問題", max_chars=500)
    if not st.button("送出"):
        return
    try:
        result = _answer(api_url, question)
    except (OSError, URLError, ValueError) as error:
        st.error(f"無法連線到 RAG 服務：{error}")
        return
    if result["status"] == "answered":
        st.write(result["answer"])
    else:
        st.warning(result["message"])
    if result["rewrite"]["status"] == "fallback":
        st.caption("查詢改寫失敗，已使用原始問題檢索。")
    elif result["rewrite"]["status"] == "rewritten":
        st.caption(f"查詢已改寫；檢索路由：{result['rewrite']['route']}。")
    st.subheader("來源")
    for source in result["sources"]:
        st.caption(
            f"{source['document_id']} · p. {', '.join(map(str, source['page_numbers']))} · {source['type']}"
        )
        st.write(source["excerpt"])


if __name__ == "__main__":
    main()
