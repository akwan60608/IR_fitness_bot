# Local RAG 的短引用代號與伺服器驗證

## 結論

本專案應保留 index 的長 `record_id` 作為真實、可稽核的 identity，但不要要求
模型重打它。每一個 answer request 在完成 retrieval 後，依最終候選順序建立短暫的
`S1`、`S2`、`S3` alias，並只把 alias 和對應 excerpt 放進 generation context。
同一份 request 的嚴格 JSON Schema 將 citation key 限為該次候選的 `enum`；伺服器
解析成功後才以 request-local map 把 alias 轉回 `record_id`、頁碼與 `source_refs`。

這是最小的完整做法：它不改 index、manifest、UI source card 或既有 evidence gate，
只取代目前「模型重現 `[source_id=<record_id>]`，再以 regex 檢查」的 generation
boundary。OpenAI 的 Structured Outputs 支援 `enum`，且 `strict: true` 會要求輸出
符合提供的 schema；JSON mode 則只保證有效 JSON，不保證 schema 遵循。
[OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)

這只解決 **reference correctness**（模型不會產生候選集合外的 ID），不是
**evidence correctness**（某個來源是否真的支持該句）。JSON Schema 是結構約束；
`enum` 的標準語意也只是值必須等於列出的其中一項。因此仍必須在 server 驗證、保留
evidence gate，並以 reviewed questions 評估 citation coverage／grounding。
[JSON Schema Validation 2020-12, `enum`](https://json-schema.org/draft/2020-12/json-schema-validation)

## 現有 Groq adapter 可直接採用

這不是需要遷移 provider 的方案：專案目前預設模型就是
`openai/gpt-oss-20b`，而鎖定的 `langchain-groq` 1.1.3 已提供
`ChatGroq.with_structured_output(schema, method="json_schema", strict=True)`。
Groq 官方文件列出此模型支援 strict mode；該模式使用 constrained decoding，要求
所有 object 欄位為 required 且 `additionalProperties: false`。因此 answer adapter
可直接從 `StrOutputParser()` 改為這個 typed result，不必換 index、供應商或新增依賴。
[Groq Structured Outputs](https://console.groq.com/docs/structured-outputs)

## 目前的切入點

目前流程在 `src/chunking/rag/service.py` 將每個 retrieved record 的完整
`record_id` 寫入 context，並接受回傳文字中的 `[source_id=...]`；
`_has_known_citations()` 只做集合 membership 與每句格式檢查。長、hash-derived
chunk ID 因此成為模型必須逐字重現的 API surface。Index 已保存 `record_id`、
`page_numbers`、`source_refs` 與 `content_sha256`，所以不需要建立第二套持久 ID。

影響範圍（尚未實作）：

- `src/chunking/rag/service.py`：建立 request-local alias map、產生 context、解析及
  驗證 structured answer、將 alias resolve 回現有 source card。
- `src/chunking/rag/api.py`：answer adapter 從 `str` 改為 schema-constrained parsed
  result；rewrite 的 typed output 不必改。
- `tests/test_rag_service.py`、`tests/test_rag_api.py`：以 alias 成功、未知 alias、空
  citation、refusal/incomplete、以及 alias 不可跨 request 重用為最小 regression cases。
- SSE response 和 Streamlit source card 可維持 public contract：仍回傳真實
  `source_id` 與現有 provenance；不把暫時 alias 當成 stable URL 或外部 API ID。

## 建議的 wire contract

retrieval 結果定序後建立 immutable map，例如：

```text
S1 -> sha256:document:0:7b3e...  (p. 2, #/texts/1)
S2 -> sha256:document:4:a91c...  (p. 4, #/texts/8)
S3 -> sha256:document:8:1540...  (p. 7, #/texts/17)
```

模型 context 只看 `Source S1`、`Source S2`、`Source S3` 和其 text。回答使用
claim-first 結構；server 依序渲染 `claims[].text`，並將 alias 展示為 UI 的 source
cards 或 inline labels：

```json
{
  "claims": [
    {"text": "深蹲時應保持背部中立。", "sources": ["S1"]}
  ]
}
```

對上例每次 request 動態產生的 schema 核心如下。`S1`/`S2`/`S3` 必須由當次
retrieval 結果產生，不能是 corpus-wide list：

```json
{
  "type": "object",
  "properties": {
    "claims": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "properties": {
          "text": {"type": "string"},
          "sources": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": ["S1", "S2", "S3"]}
          }
        },
        "required": ["text", "sources"],
        "additionalProperties": false
      }
    }
  },
  "required": ["claims"],
  "additionalProperties": false
}
```

在本專案的 Groq adapter，將此 schema 傳給
`llm.with_structured_output(schema, method="json_schema", strict=True)`。strict mode
要求每個 object 的 `additionalProperties: false` 與所有欄位列入 `required`；不符合
限制的 schema 會被 API 拒絕。[Groq Structured Outputs](https://console.groq.com/docs/structured-outputs)

若 adapter 已經必須走 function-calling loop，等價替代是嚴格的 `cite_answer`
function，其 `sources.items.enum` 同樣由候選 alias 產生。這並不比 response schema
更適合本服務：function calling 需要 application 執行 tool call 後再送 tool output
取得最終文字，是額外一輪協調；OpenAI 也將它定義為多步 flow。只有 citation 本身
需要 server-side action 時才選它。
[OpenAI Function calling flow](https://developers.openai.com/api/docs/guides/function-calling)

## 必要的伺服器端規則

1. alias 僅是本次 `request_id` 內的 lookup key。以最終排序給它固定名稱，map 保持
   在 `RagService.answer()` 的 local scope；不可接受 client 回傳的 map，也不可把
   alias 存進 index、cache key 或長期 URL。
2. 對每個 claim，拒絕空文字、空 sources、重複 alias，及不在 map 的 alias；resolve
   後只從 server-side record 取 `record_id`、page、excerpt、hash 和 source refs。
   Schema 的 enum 已先消除正常模型輸出中的未知 key，但這項檢查仍是 trust boundary。
3. 將 refusal、`incomplete` response、parse/schema error 和不存在有效 citation 視為
   `citation_validation_failed`（或既有 generation failure），只回傳 diagnostic
   sources，不要輸出未驗證回答。Structured Outputs 的官方文件明確要求 callers 處理
   refusal；嚴格 schema 的保證也不涵蓋被中斷的 generation。
   [OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
4. 不要把「alias 存在」誤報成「claim 被證實」。它只證明該 citation 指向本次
   retrieved record；保留目前 evidence gate，並新增 human-reviewed assertion：每個
   rendered claim 至少有一個可支持它的 resolved record。這是從 JSON Schema 的結構
   驗證範圍所作的設計推論，而非 schema 可自動提供的保證。

`top_k=3` 的現況非常適合 enum。若未來把大量候選同時交給模型，schema 本身有上限：
最多 1,000 個 enum value，且 enum 字串大小亦有限制；應先縮小 generation candidates，
不能把完整 corpus 放進 enum。[Structured Outputs schema limits](https://developers.openai.com/api/docs/guides/structured-outputs)

## 方案比較

| 方案 | 可驗證的事 | 成本／限制 | 判定 |
| --- | --- | --- | --- |
| 短 alias + strict response-schema enum | key 必定屬於本次候選；server 可無歧義映射長 ID | adapter 改為 structured parse；仍需檢查 grounding | **建議**；符合本機 index 和現有 top-3 flow |
| 短 alias + 現有 inline regex | server 可驗證 alias membership | prompt/regex 仍要處理格式、漏引和句子邊界；沒有 constrained decoding | 可作過渡，但不應是最終 contract |
| 直接要求長 `record_id` | server 可做 membership | 模型需逐字複製長、內部 ID；正是本問題 | 不建議 |
| 全域固定短 ID | 易於顯示 | index rebuild、document replacement 或多 index 容易造成 stale/wrong mapping；要增加 versioning | 除非外部 permalink 是需求，否則不建 |
| provider File Search citations | provider output 會給 file citation annotation 和 `file_id` | 需要搬遷／同步本機 corpus；citation 是 file-level，不等於本機 chunk/page evidence | 僅在要採用 provider-hosted retrieval 時考慮 |

## Provider-hosted File Search 的界線

OpenAI Responses 的 File Search 在 tool 被呼叫時回傳 `file_search_call` 和帶
`file_citation` annotations 的 message；官方 example 的 annotation 含 `file_id`、
`filename` 和 text insertion `index`。因此 renderer 應保留 annotations，而不只取
flattened text；example 也顯示同一 citation 可能重複，display 時可 dedupe。
[OpenAI File Search response example](https://developers.openai.com/api/docs/guides/tools-file-search)

Search result records 預設不會回傳；若要同時取得 provider retrieval evidence，request
要加 `include: ["file_search_call.results"]`。
[OpenAI File Search include results](https://developers.openai.com/api/docs/guides/tools-file-search)

這可免去模型生成 local ID，但不是本機 index 的 drop-in citation validator：官方
`file_citation` example 是 file-level `file_id`/filename，沒有本專案要的原始 chunk
`record_id`、page 或 `source_refs`。採用它時仍要在 upload 時維護
`provider file_id -> local document/index revision` 的 server-side manifest，並把
file-level citation 明確標示為 file provenance，而非聲稱它驗證到 chunk 或 claim。

## 實作順序與驗收

1. 在 service 的 retrieval-to-generation boundary 建 request-local alias map，先保留
   SSE／Streamlit 的真實 source card payload。
2. 讓 answer adapter 回傳上述 typed claims；不要在 prompt 內再暴露長 ID。
3. server parse、validate、resolve 後才組 answer；任何失敗沿用 diagnostic-source
   failure path。
4. 最小 tests：長 ID 永不出現在 model context；`S1` resolve 成原 record；未知或空
   citation 拒絕；同樣 `S1` 在兩個 request 不能指到第二次的舊 record；refusal／
   incomplete 不可成為 answered。

未做：本研究沒有改應用程式、測試 provider SDK 的即時行為，或主張 schema 能評估
語意 grounding。風險：若改用 provider File Search，資料會離開本機流程，且其
file-level citations 不足以取代本專案的 chunk/page provenance。
