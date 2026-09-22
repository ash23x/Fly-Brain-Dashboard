"""Quick census: which named circuits exist in the MaleCNS annotations?

Central complex compass (EPG / PEN / PEG / Delta7), mushroom body
(Kenyon cells, MBONs, dopamine neurons) and motion detectors (T4/T5).
Read-only. Run: python scripts/probe_circuits.py
"""
import pathlib, re
import pandas as pd, pyarrow.feather as feather

ROOT = pathlib.Path(__file__).resolve().parent.parent
p = ROOT / "data" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
cols = ["bodyId", "type", "instance", "class", "superclass", "somaSide", "hemibrainType", "group"]
df = feather.read_table(str(p), columns=cols).to_pandas()
t = df["type"].fillna("").astype(str)

PATTERNS = {
    "EPG": r"^EPG", "PEN": r"^PEN", "PEG": r"^PEG", "Delta7": r"^Delta7",
    "EL": r"^EL$", "ER (ring)": r"^ER\d", "ExR": r"^ExR",
    "KC": r"^KC", "MBON": r"^MBON", "PAM": r"^PAM", "PPL1": r"^PPL1",
    "T4": r"^T4", "T5": r"^T5", "LPTC/HS/VS": r"^(HS|VS|H2|CH)",
    "DNa (descending steer)": r"^DNa", "DNp": r"^DNp",
}
for name, pat in PATTERNS.items():
    m = t.str.contains(pat, regex=True)
    types = sorted(t[m].unique())
    print(f"{name:24s} {int(m.sum()):6d} neurons  {len(types):4d} types  e.g. {types[:8]}")

print("\nsuperclass counts:")
print(df["superclass"].value_counts().head(25).to_string())
print("\nEPG instances sample:")
print(df.loc[t.str.match(r"^EPG"), ["type", "instance", "somaSide", "group"]].head(12).to_string())
print("\nPEN instances sample:")
print(df.loc[t.str.match(r"^PEN"), ["type", "instance", "somaSide"]].head(8).to_string())
