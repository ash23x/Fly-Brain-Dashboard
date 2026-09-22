# The data

This project runs on the **MaleCNS v1.0** connectome — the complete central
nervous system of one adult male *Drosophila melanogaster*, reconstructed
from electron microscopy by HHMI Janelia FlyEM with the University of
Cambridge, MRC LMB and Google Research, and released in September 2026.

- Download page: https://male-cns.janelia.org/download/
- Files used here (fetched once by `scripts/01_download_data.py` from the
  bucket that page lists): the body annotations, the predicted
  neurotransmitters, and the "significant connections" synapse-count table.
- Licence: **CC-BY 4.0**. If you use this project or anything derived from
  it, credit the MaleCNS release and its authors.

The data is not in this repository (it is ~560 MB and gitignored). The
code here is a separate work and carries its own licence file.
