# Restricted resources

This repository contains implementation code and artificial test inputs only.
It does not distribute patient records, de-identified clinical narratives,
reference labels, case-level predictions, cohort partitions, retrieval cases,
embeddings, vector indexes, proprietary knowledge cards, source books or papers,
model weights, credentials, request bodies, responses, or execution logs.

Patient-level resources remain restricted by patient privacy and institutional
data-use requirements. Knowledge-base contents are withheld to protect copyright.
Public availability of this code does not grant access to, or a license for,
those resources. No patient data should be posted in issues or pull requests.

## Local use

Supply authorized resources through the interfaces described in the README.
Keep those resources outside this repository. Complete de-identification and
free-text review before any authorized model processing. The reference interfaces
are not a complete de-identification tool and cannot establish compliance by
themselves. Do not place credentials in source code or configuration committed
to version control.

The tests run locally without network access, credentials, model downloads, or
clinical data. Toy evidence identifiers and toy class names in the tests are
artificial and are not derived from patient records or knowledge-base content.

## Publication boundary

This is a standalone implementation reference, not an export of the original
experiment repository or its history. Study-specific private resources and
archived execution environments are not included. Tests establish selected
software invariants; they do not reproduce clinical performance or establish
clinical safety. The code is for research inspection, not patient care.
