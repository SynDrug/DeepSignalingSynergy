# DeepSiganlingSynergy
## Drug Combo Predictions on Non-Fully Connected DNN

* Get Information About How Many Lines Of Code
```
find . -name '*.py' | xargs wc -l
```

### Environment Setting
* Mac OS Catalina Version 10.15.6
* Packages: 
    * tensorflow 1.13.1
    * sklearn
    * innvestigate

### Run 4features_model with no Gene and Pathway Analysis
* The model built with sequential
* In file '4features_model', compile python3 main.py

### Run Model with Gene and Pathway Analysis
```
python3 parse_file.py
python3 load_data.py
python3 main.py
```