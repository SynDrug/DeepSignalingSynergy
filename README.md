# DeepSignalingSynergy

In this study, we proposed a not fully connected deep learning model, DeepSignalingSynergy, for drug combination prediction. Compared with existing models that use a large number of chemical-structure and genomics features in densely connected layers, we built the model on a small set of cancer signaling pathways, which can mimic the integration of multi-omics data and drug target/mechanism in a more biological meaningful and explainable manner. 

<!-- For more details of DeepSignalingSynergy, see our [bioRxiv paper]() -->

### Dependencies
* python 3.7.3
* tensorflow 1.13.1
* pandas
* sklearn

## 1.Data Preprocess
This study intergrates following datasets
* NCI ALMANAC drug combination screening dataset 
* Gene expression data of NCI-60 Cancer Cell Lines
* KEGG signaling pathways and cellular process
* Drug-Target interactions from DrugBank database

Finally, those datasets files will be parsed into numpy files to train our DeepSignalingSynergy model.

```
python3 parse_file.py
python3 load_data.py
```

## 2.Running DeepSignalingSynergy
Run the code
```
python3 main.py
```
Analyze the experiment results and plot figures
```
python3 analysis.py
```

