# Historical GALVAT notebook archive

These notebooks were imported on2026-11-02 from `domains/galvat` in the ML workbench. Original subdirectory names are retained. Saved outputs, execution counts and transient metadata were cleared before publication. Source cells are unchanged except that one embedded API-key literal in Trial5 was replaced by an interactive `getpass` prompt. The original workbench notebook was not modified; rotate that key if still active. Exact input/output hashes are in [the import manifest](../../docs/import_manifest.json).

These are **historical snapshots, not certified standalone workflows or current model provenance**. Some require supporting workbench scripts, external data, Colab secrets and environment setup that this import does not migrate. Importing them does not execute training, verify metrics or install dependencies.

In particular, `trial_9/Trial_9_Train_Evaluate_Export.ipynb` explicitly expects **44 classes**. It predates the current45-class model and does not include measuring spoon. Do not use it as the current training/export notebook. The supplied [small-model Colab source](../current/README.md) is now available separately; the nano source remains outstanding.

Do not patch this archived notebook to45 classes and present that as the original training source. Once supplied, add the actual current source under `notebooks/current/` with its model/evaluation manifest. Keep historical trials for traceability.
