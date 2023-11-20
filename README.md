# NormFAAE

NormFAAE: A Filter-Augmented Auto-Encoder with Learnable Normalization for Robust Multivariate Time Series Anomaly Detection

The main contributions of NormFAAE are as follows:
  1. A filter-augmented auto-encoder framework with learnable normalization is proposed.
  2. A deep hybrid normalization module is proposed.
  3. A filter-augmented auto-encoder model is proposed.

![image](https://github.com/MachineLearning921/NormFAAE/assets/151547001/3b345d25-824f-49f5-964e-fb6f368071a0)

## Get Started

1. Requirements: Python 3.8, PyTorch 1.12. 
2. Download data. You can obtain four benchmarks from [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.
3. Train and evaluate. You can reproduce the experiment results as follows:
```bash
python main.py --data 'SWAT'
python main.py --data 'SMD'
python main.py --data 'PSM'
python main.py --data 'MSL'
python main.py --data 'SMAP'
```

## Main Result
We compare our model with 17 baselines. **Generally,  NormFAAE achieves SOTA.**

![image](https://github.com/MachineLearning921/NormFAAE/assets/151547001/95b4adf8-4c0a-4d30-8478-ca70f8434361)
![image](https://github.com/MachineLearning921/NormFAAE/assets/151547001/92c416f6-acda-43c1-97f6-495986a99e9a)
![image](https://github.com/MachineLearning921/NormFAAE/assets/151547001/cceee55e-9e49-4354-add7-ae8a1ca68ce5)


## Citation
If you find this repo useful, please cite our paper. 

```

```
