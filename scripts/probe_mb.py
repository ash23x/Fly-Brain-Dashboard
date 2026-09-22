"""Census of the mushroom-body learning circuit in the MaleCNS annotations."""
import pathlib, re, collections
import pandas as pd, pyarrow.feather as feather
ROOT = pathlib.Path(__file__).resolve().parent.parent
p = ROOT / "data" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
df = feather.read_table(str(p), columns=["bodyId", "type", "instance", "class", "superclass", "somaSide"]).to_pandas()
t = df["type"].fillna("").astype(str)
def show(name, mask, n=12):
    types = collections.Counter(t[mask])
    print(f"{name:34s} {int(mask.sum()):6d} neurons {len(types):4d} types  e.g. {[k for k, _ in types.most_common(n)]}")
show("KC (Kenyon cells)", t.str.match(r"^KC"))
show("MBON", t.str.match(r"^MBON"))
show("PAM (reward DANs)", t.str.match(r"^PAM"))
show("PPL1 (punish DANs)", t.str.match(r"^PPL1"))
show("APL / DPM", t.isin(["APL", "DPM"]))
show("uniglomerular PNs (_adPN/_lPN/_ilPN/_lvPN)", t.str.contains(r"_(ad|l|il|lv)PN", regex=True))
show("multiglomerular PNs (M_)", t.str.match(r"^M_.*PN"))
show("any type containing PN", t.str.contains(r"PN"))
print("\nsample PN types:", sorted(set(t[t.str.contains(r"_(ad|l|il|lv)PN", regex=True)]))[:40])
print("\nKC types:", collections.Counter(t[t.str.match(r"^KC")]).most_common())
print("\nMBON types:", sorted(set(t[t.str.match(r"^MBON")])))
