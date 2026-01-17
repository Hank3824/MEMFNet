# MEMFNet

<img src='/git-memfnet.png' class="center" width="99%">
 
A Python package for interpretable prediction of battery discharge profiles incorporating composition and synthesis conditions in high-nickel layered cathodes.

Paper at: [https://doi.org/10.1016/j.nanoen.2026.111735](https://doi.org/10.1016/j.nanoen.2026.111735) 




**Inputs**:

1. **composition** (string) - Chemical composition formula
2. **V_low** (float) - Lower voltage limit (V)
3. **V_high** (float) - Upper voltage limit (V)
4. **rate** (float) - Current density rate (mA/g)
5. **cycle** (int) - Cycle number
6. **Vii** (float) - Current voltage point (V)
7. **sin1_temp** (float) - First sintering temperature (°C)
8. **sin1_time** (float) - First sintering time (h)
9. **sin2_temp** (float) - Second sintering temperature (°C)
10. **sin2_time** (float) - Second sintering time (h)
11. **sin2_exists** (int) - Whether second sintering exists (0/1)

We currently provide a partial dataset to facilitate quick testing and demonstration. The complete dataset will be made available upon request.

## Dependency
* Requirements: `pymatgen`, `torch`, `torch-scatter`.
* Here is a simple installation if you are using `MEMFNet` for prediction on GPU:

```sh
conda create -n memfnet python=3.9
conda activate memfnet

pip install pymatgen
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

## Installation
```sh
python setup.py install
pip install -e .
```


## Acknowledgement

The [Roost](https://github.com/CompRhys/roost) (Representation Learning from Stoichiometry) and [mat2vec](https://github.com/materialsintelligence/mat2vec) is used for compositional encoding in MEMFNet. Please consider citing the relevant works:


* R. E. A. Goodall and A. A. Lee, Predicting Materials Properties without Crystal Structure: Deep Representation Learning from Stoichiometry, Nat. Commun. 11, 6280 (2020). [[link]](http://www.nature.com/articles/s41467-020-19964-7)

* V. Tshitoyan, J. Dagdelen, L. Weston, A. Dunn, Z. Rong, O. Kononova, K. A. Persson, G. Ceder, and A. Jain, Unsupervised Word Embeddings Capture Latent Knowledge from Materials Science Literature, Nature 571, 95 (2019). [[link]](http://www.nature.com/articles/s41586-019-1335-8)

* P. Zhong, et al., Deep learning of experimental electrochemistry for battery cathodes across diverse compositions. Joule (2024). [[link]](https://doi.org/10.1016/j.joule.2024.03.010) [[code]](https://github.com/zhongpc/drxnet)
