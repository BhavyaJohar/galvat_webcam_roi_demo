# Current supplied training notebooks

`GALVAT_Trial9_Train_Evaluate_Package_small.ipynb` is Bhavya's supplied small-model Colab notebook, received with the November2 delivery-plan update. It uses YOLO26s, includes a measuring-spoon GALVAT rating and asserts45 rated classes. The publication copy preserves source cells while clearing outputs, execution counts and transient metadata. No training or evaluation was rerun.

## Interpretation and verification still required

- Some markdown headings and output keys still refer to `shared44`, although the active rating code asserts45 classes. Preserve this source history; reconcile metric names and definitions before the release report rather than treating those labels as proof of44-class evaluation.
- The source starts with evaluation `MAX_DET = 300` and later exports with `max_det=40`. A matched300-versus40 export comparison remains necessary before claiming unchanged recall or a quantified device benefit from that limit.
- Saved source-notebook outputs report validation GALVAT0.7478051809038424 and frozen-test GALVAT0.6920212209535992. These are historical notebook outputs, not independently reproduced results or nano-model metrics. Publication outputs are cleared; retain the original export privately as execution evidence.
- Confirm exact selected checkpoint, canonical class-ID order, input/output contract, ONNX hash and per-class thresholds against the deployment bundle. Presence of45 classes does not establish ordering or binary provenance.
- The nano training/evaluation notebook is still outstanding. The historical44-class Trial9 template remains separately archived and is not the current training source.

Provenance is recorded in [the supplied notebook manifest](../../docs/supplied_small_notebook.json). Model building and evaluation documentation belong in this repo; Unity application changes belong in `jadinV/GALVAT_R61`.
