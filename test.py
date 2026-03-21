from safetensors import safe_open

with safe_open("train/FoldPep/res_model/FoldPep_2000/model.safetensors", framework="pt") as f:
    print(list(f.keys()))