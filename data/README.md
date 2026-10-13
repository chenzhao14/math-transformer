# Generated arithmetic data

No original datasets are bundled. `tokenization.ipynb` generates and tokenizes
addition data for operand ranges 10, 30, 100, and 300. `train_eval.py` generates
its own range-30 data. Subtraction examples can be created with
`python generate_subtraction_datasets.py`.

The original `split_train_val` function intentionally includes validation records
in the training set. Its validation accuracy therefore measures seen-example
performance, not unseen-example generalization. Random sampling can also repeat
operand pairs. Both behaviors are retained and documented for this release.
