# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

import os

# Minimum uncached prompt tokens before disaggregated prefill routes to a linked
# remote prefill instance instead of running prefill locally. Below this, the
# round-trip isn't worth it. Overridable for testing shorter prompts against a
# real remote-prefill path without needing 1000+ token pastes each time.
REMOTE_PREFILL_MIN_TOKENS: int = int(os.getenv("EXO_REMOTE_PREFILL_MIN_TOKENS", "1000"))

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
