# Implementation For OMG Framework

## 1. Train model

```python
cd train
python train.py -l1 1.5 -l2 1 -l3 0.5 -t 1.0
```

## 2. Inference model

```python
cd generate
python generate.py --model_weight <your_model> --csv_name result
```

