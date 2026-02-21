import json

import rlcard

env = rlcard.make("no-limit-holdem")
state, player_id = env.reset()

print("player_id:", player_id)
print("state keys:", list(state.keys()))

raw_obs = state.get("raw_obs")
print("has raw_obs:", raw_obs is not None)
if isinstance(raw_obs, dict):
    print("raw_obs keys:", list(raw_obs.keys()))
    sample_keys = list(raw_obs.keys())[:10]
    sample = {k: raw_obs.get(k) for k in sample_keys}
    print("raw_obs sample:", json.dumps(sample, default=str)[:800])

legal = state.get("legal_actions")
if hasattr(legal, "keys"):
    legal_ids = list(legal.keys())
else:
    legal_ids = list(legal)
print("legal_action_ids:", legal_ids)
