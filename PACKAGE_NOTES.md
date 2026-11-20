# Packaging and provenance notes

This GitHub-ready repository was assembled from two user-provided files:

- `Bladder_EIS_IJS_Full_Reproduction (1).ipynb` — newer, expanded, 78-cell notebook; repackaged as `Bladder_EIS_IJS_Full_Reproduction.ipynb` without modifying its cell sources.
- `Iranian_Journal_of_Science_25pages_1.zip` — source modules, `results/`, `figs/`, MIT license and citation metadata originally located under `Iranian_Journal_of_Science/code_for_repository/`.

`article_tables.py` was not present as a stand-alone Python file in the older `code_for_repository/` subfolder; its source was extracted exactly from the supplied newer notebook's `%%writefile article_tables.py` cell.

New helper files (`prepare_cached_results.py`, `verify_repository.py`), Git metadata settings, requirements and this README were created solely to make the repository easier to use and publish. The original scientific analysis routines were not rewritten.

The original ZIP also held manuscript PDFs/TeX, author/editorial correspondence and internal review documents, which were deliberately excluded from the public-code package. The separately hosted source dataset is also excluded. Other than this intentional selection, the original Python modules, saved results and five manuscript figures are retained.
