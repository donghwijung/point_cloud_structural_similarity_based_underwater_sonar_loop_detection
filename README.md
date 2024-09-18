# Point Cloud Structural Similarity-based Underwater Sonar Loop Detection

## Description
This is an implentation of *"Point cloud structural similarity-based underwater sonar loop detection"* which indicates detecting loops based on the structural similarity of point clouds generated from the data acquired by MBES.

## Dataset
- Download datasets [link](https://drive.google.com/drive/folders/1MV_GaNRxmcbjUQT7r6kNH1NtUX6jMK07?usp=sharing)
- Unzip the downloaded dataset

## Installation
```
conda env create -f environment.yaml
```

## Execution
```
conda activate PCSS
python execute.py
# ex) python execute.py --data_path data --data_id 1 --neighborhood_size 100 --score_threshold 2.95
```

## Acknowledgements
The code in this repository is based on [PointSSIM](https://github.com/mmspg/pointssim). Thanks to the authors of the code.