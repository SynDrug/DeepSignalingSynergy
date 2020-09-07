# DeepSignalingSynergy

In this study, we proposed a not fully connected deep learning model, DeepSignalingSynergy, for drug combination prediction. Compared with existing models that use a large number of chemical-structure and genomics features in densely connected layers, we built the model on a small set of cancer signaling pathways, which can mimic the integration of multi-omics data and drug target/mechanism in a more biological meaningful and explainable manner. 

<!-- For more details of DeepSignaling, see our [bioRxiv paper]() -->

### Environment Setting
* python 3.7.3
* tensorflow 1.13.1
* pandas
* sklearn
* innvestigate

### Data Preprocess
We need to use pandas to parse files into a standard version of numpy files.
```
python3 parse_file.py
python3 load_data.py
```

### Running DeepSignalingSynergy
Run the code
```
python3 main.py
```
Analyze the experiment results and plot figures
```
python3 analysis.py
```

