# Third-Party Notices

The public algorithms and solver interfaces in this repository are named
**CP_AL** and **FA_CP**. Copyright in the algorithmic modifications,
experiment policies, release scripts, and documentation is held by Benqi Liu.

Some low-level GPU, sparse linear-algebra, QP input, and solver-interface
components were adapted from Apache-2.0 software. Redistribution therefore
retains the original notices and records modifications separately:

- LP infrastructure was adapted from cuPDLPx v0.2.9
  ([MIT-Lu-Lab/cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx), revision
  `931c94c0c3767d38b3df71514c84ccd314aa2ac7`), whose source files state
  Copyright 2025 Haihao Lu.
- QP infrastructure was adapted from PDHCG-II v0.1.3
  ([Lhongpei/PDHCG-II](https://github.com/Lhongpei/PDHCG-II), revision
  `ee04fd9431c59172b9107567f7c293536272fabc`), whose source files retain
  notices for Hongpei Li and Haihao Lu.

Files changed for this release carry a prominent notice of modification by
Benqi Liu. The retained names and copyrights identify the origin of low-level
components; they do not rename, co-author, or endorse the CP_AL and FA_CP
research methods.

HPR-LP-C and the other solvers named under `experiments/lp_benchmarks` are
external comparison programs. Their source and binaries are not distributed
in this repository.
