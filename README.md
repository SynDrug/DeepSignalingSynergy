# DeepSiganlingSynergy
## Drug Combo Predictions on Non-Fully Connected DNN

* Get Information About How Many Lines Of Code
```
find . -name '*.py' | xargs wc -l
```

### Run Origin Model with no Gene and Pathway Analysis
* The model built with sequential
* In file 'origin_model', compile python3 main.py

### Run Model with Gene and Pathway Analysis
```
python3 parse_file.py
python3 load_data.py
python3 main.py
```