import json

notebook_path = "/mnt/12bc61a2-ddc5-4be8-b8be-bbd99dc6d141/AI engineer/Study/Speech-Emotions-Recognition/SER.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Number of cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    source = cell.get('source', [])
    source_str = "".join(source)
    first_few_lines = "\n".join(source_str.split("\n")[:3])
    if cell_type == "markdown":
        print(f"Cell {idx} (Markdown): {first_few_lines}")
    elif cell_type == "code":
        # Check if it has a header comment
        lines = source_str.split("\n")
        header = ""
        for line in lines:
            if line.strip().startswith("#"):
                header = line.strip()
                break
        print(f"Cell {idx} (Code): {header[:80]} (lines: {len(lines)})")
