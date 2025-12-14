import time

class Metrics:
    def __init__(self):
        self.start = time.time()
        self.llm_calls = 0
        self.tokens = 0

    def record_llm_call(self, prompt: str, response: str):
        self.llm_calls += 1
        self.tokens += len(prompt.split()) + len(response.split())

    def finalize(self):
        return {
            "latency_ms": int((time.time() - self.start) * 1000),
            "llm_calls": self.llm_calls,
            "approx_tokens": self.tokens
        }
