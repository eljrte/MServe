def print_kv(past_key_values):
    for i, (k, v) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
        print(f"Layer {i}: key shape = {k.shape}, value shape = {v.shape}")