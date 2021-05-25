**EnMUGR for Biological Networks Representation**

EnMUGR is a method to learn node representations for biological networks. It focuses on extracting effective and consistent information from heterogeneous networks. Low-dimensional representation features learned by the method can be used in general network-based inference problems.

The _run_demo.m_ provides an example for learning representation features with EnMUGR.
The _run_simulation.m_ provides a full case for running EnMUGR.

**Dependencies**

[Network enhancement](http://snap.stanford.edu/ne/)

[L-BFGS-B](https://www.mathworks.com/matlabcentral/fileexchange/35104-lbfgsb--l-bfgs-b--mex-wrapper)

**Datasets**

Datasets used in EnMUGR are downloaded from following websites:

[Drug and target networks](https://github.com/luoyunan/DTINet)

[Human protein networks](http://cb.csail.mit.edu/cb/mashup/)

[Butterfly species similarity networks](http://snap.stanford.edu/ne/)

The datasets used in toy example and simulation study were generated according to [1].

**Reference**

[1]Chauvel C, et al. Evaluation of integrative clustering methods for the analysis of multi-omics data. Briefings in Bioinformatics, 21(2):541–552, 2020.
