# claude_batch_client.py
# pip install -U anthropic
# export ANTHROPIC_API_KEY=...

import os, time, itertools
from typing import List, Dict, Union, Iterable, Optional, Tuple
import anthropic
from dotenv import load_dotenv
import ipdb
load_dotenv()

MODEL_DEFAULT = "claude-3-5-haiku-20241022"  # 或 "claude-haiku-4-5-20251001"


class GenerateOutput:
    def __init__(self, text: List[str], raw: Optional[Dict]=None, usage: Optional[List[Dict]]=None):
        # text: 對應該樣本的 N 個候選（batch 以重複請求實現）
        self.text = text
        self.n = len(text)
        self.raw = raw  # 可選：保留原始映射資訊 (custom_id -> 原始 entry)
        self.usage = usage  # 新增：每個候選的 token 使用統計
    
    

class ClaudeBatchModel:
    def __init__(
        self,
        model: str = MODEL_DEFAULT,
        temperature: float = 1.0,
        max_tokens: int = 10000,
        poll_sec: int = 10,
    ):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.poll_sec = poll_sec

    # ----------------- public API -----------------
    def generate(
        self,
        prompt: Union[str, List[str]],
        num_return_sequences: int = 1,
    ) -> Union[GenerateOutput, List[GenerateOutput]]:
        """
        - prompt: str -> 回傳單一 GenerateOutput（包含 num_return_sequences 個候選）
        - prompt: List[str] -> 回傳 List[GenerateOutput]（順序與輸入對齊）
        """
        if isinstance(prompt, str):
            prompts = [prompt]
            per_item = num_return_sequences
            single = True
        else:
            prompts = list(prompt)
            per_item = num_return_sequences
            single = False

        # 1) 建立 batch 請求（為了拿到 N 候選，每個樣本提交 N 份請求）
        requests = []
        # custom_id 形如: s{sample_idx:06d}_n{variant_idx:02d}
        for i, p in enumerate(prompts):
            for k in range(per_item):
                requests.append({
                    "custom_id": f"s{i:06d}_n{k:02d}",
                    "params": {
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        # "temperature": self.temperature,
                        "messages": [{"role": "user", "content": p}],
                    },
                })

        batch = self.client.messages.batches.create(requests=requests)
        batch_id = batch.id

        # 2) 等待處理完成
        self._wait_for_batch(batch_id)

        # 3) 收集結果（custom_id -> text）
        cid2text, cid2raw, cid2usage = self._collect_results(batch_id)

        # 4) 組裝回傳：對每個樣本，依 n=0..per_item-1 的 custom_id 順序取回
        outputs: List[GenerateOutput] = []
        for i in range(len(prompts)):
            texts = []
            raw_map = {}
            usage_list = []
            for k in range(per_item):
                cid = f"s{i:06d}_n{k:02d}"
                texts.append(cid2text.get(cid, ""))  # 若失敗則給空字串
                raw_map[cid] = cid2raw.get(cid)
                usage_list.append(cid2usage.get(cid))
            outputs.append(GenerateOutput(text=texts, raw=raw_map, usage=usage_list))

        return outputs[0] if single else outputs
        

    # ----------------- internals -----------------
    def _wait_for_batch(self, batch_id: str) -> None:
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                return
            time.sleep(self.poll_sec)

    def _extract_text_from_message(self, msg_obj) -> str:
        """
        msg_obj: anthropic.types.Message 或等價 dict
        將所有 text block 串起來。
        """
        content = getattr(msg_obj, "content", None)
        if content is None and isinstance(msg_obj, dict):
            content = msg_obj.get("content", [])
        text_parts = []
        if isinstance(content, Iterable):
            for b in content:
                b_type = getattr(b, "type", None) if not isinstance(b, dict) else b.get("type")
                if b_type == "text":
                    t = getattr(b, "text", "") if not isinstance(b, dict) else b.get("text", "")
                    if t:
                        text_parts.append(t)
        return "".join(text_parts).strip()

    def _collect_results(self, batch_id: str) -> Tuple[Dict[str, str], Dict[str, dict], Dict[str, dict]]:
        """
        讀取批次結果：
        - 回傳三個 dict：
        (custom_id -> text), (custom_id -> 原始 entry), (custom_id -> usage)
        """
        cid2text: Dict[str, str] = {}
        cid2raw: Dict[str, dict] = {}
        cid2usage: Dict[str, dict] = {}

        for entry in self.client.messages.batches.results(batch_id):
            # entry 可能是 Pydantic 物件
            cid = getattr(entry, "custom_id", None) or getattr(entry, "id", None)
            if cid is None and isinstance(entry, dict):
                cid = entry.get("custom_id") or entry.get("id")

            result = getattr(entry, "result", None)
            if result is None and isinstance(entry, dict):
                result = entry.get("result")

            rtype = getattr(result, "type", None) if result is not None else None
            if rtype is None and isinstance(result, dict):
                rtype = result.get("type")

            if rtype == "succeeded":
                msg_obj = getattr(result, "message", None)
                if msg_obj is None and isinstance(result, dict):
                    msg_obj = result.get("message", {})
                text = self._extract_text_from_message(msg_obj)
                cid2text[cid] = text
                cid2raw[cid] = {"ok": True}
                # 提取 usage 信息
                usage = getattr(msg_obj, "usage", None)
                if usage is None and isinstance(msg_obj, dict):
                    usage = msg_obj.get("usage", {})
                
                if usage:
                    print(usage)
                    cid2usage[cid] = {
                        "input_tokens": getattr(usage, "input_tokens", 0) if not isinstance(usage, dict) else usage.get("input_tokens", 0),
                        "output_tokens": getattr(usage, "output_tokens", 0) if not isinstance(usage, dict) else usage.get("output_tokens", 0),
                    }
                else:
                    cid2usage[cid] = {"input_tokens": 0, "output_tokens": 0}
            else:
                # 盡量保留錯誤資訊
                cid2text[cid] = ""
                cid2raw[cid] = {"ok": False, "result": repr(result)}
                cid2usage[cid] = {"input_tokens": 0, "output_tokens": 0}

        return cid2text, cid2raw, cid2usage


# ------------- minimal usage -------------
if __name__ == "__main__":
    import json
    fp = "data/movie/benchmark/2025-07-01_2025-09-30/en/sampled_250/stance_dist.jsonl"
    data = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    qs = []
    for i in range(10):
        qs.append(data[i]["question"])
    

    m = ClaudeBatchModel(
        model=MODEL_DEFAULT,
        temperature=0.0,
        max_tokens=512,
        poll_sec=10,
    )

    # 1) 單一樣本、取 2 個候選
    # out = m.generate("Say this is a test.", num_return_sequences=2)
    # print("[single] n=", out.n, " -> ", out.text)

    # 2) 多樣本、每個取 1 個候選
    outs = m.generate(
        qs,
        num_return_sequences=1,
    )
    for i, r in enumerate(outs, 1):
        print(f"[{i}] n={r.n} -> {r.text[0]}")
