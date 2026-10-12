"""Converts hierarchical_customer_clustering.py (with '# %%' markers) into a .ipynb"""
import re, nbformat as nbf
src = open("hierarchical_customer_clustering.py").read()
parts = re.split(r"^# %%(.*)$", src, flags=re.M)
nb = nbf.v4.new_notebook()
for header, body in zip(parts[1::2], parts[2::2]):
    body = body.strip("\n")
    if "[markdown]" in header:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        nb.cells.append(nbf.v4.new_markdown_cell(text))
    else:
        nb.cells.append(nbf.v4.new_code_cell(body))
nbf.write(nb, "hierarchical_customer_clustering.ipynb")
print("Notebook created.")
