import json
import os

notebook_path = "/mnt/12bc61a2-ddc5-4be8-b8be-bbd99dc6d141/AI engineer/Study/Speech-Emotions-Recognition/SER.ipynb"
out_dir = "/mnt/12bc61a2-ddc5-4be8-b8be-bbd99dc6d141/AI engineer/Study/Speech-Emotions-Recognition/exported_cells"
os.makedirs(out_dir, exist_ok=True)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    if cell_type == "code":
        source = cell.get('source', [])
        source_str = "".join(source)
        # Find first non-empty line to name the file
        name = f"cell_{idx}"
        for line in source_str.split("\n"):
            cleaned = line.strip().replace("=", "").replace("-", "").replace("#", "").strip()
            if cleaned:
                # keep only alphanumeric and underscore
                safe_name = "".join(c if c.isalnum() or c=='_' else '_' for c in cleaned)
                safe_name = safe_name.strip("_")[:50]
                if safe_name:
                    name = f"cell_{idx}_{safe_name}"
                    break
        out_path = os.path.join(out_dir, f"{name}.py")
        with open(out_path, 'w', encoding='utf-8') as out_f:
            out_f.write(source_str)
        print(f"Exported {out_path}")
