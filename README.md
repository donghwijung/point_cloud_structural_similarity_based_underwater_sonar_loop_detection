# Point Cloud Structural Similarity-based Underwater Sonar Loop Detection

## Description
This is an implentation of *"Point cloud structural similarity-based underwater sonar loop detection"* which indicates detecting loops based on the structural similarity of point clouds generated from the data acquired by MBES.

## Installation
```bash
conda env create -f environment.yaml
conda activate PCSS
```

## Dataset
### Download
- Download datasets [link](https://drive.google.com/drive/folders/1MV_GaNRxmcbjUQT7r6kNH1NtUX6jMK07?usp=sharing)
- Unzip the downloaded dataset
### Data processing
If you would like to generate the data yourself, follow the process outlined below.
To process the Antarctica dataset, the *auv_lib* library must be installed. The *antarctica_2019.cereal* file is also required. You can download the file [here](https://drive.google.com/drive/folders/1UWxJw6cNCvzowqWpzo5eSEUT_0734tsG).
```bash
python generate_data_from_antarctica.py
```
or
```bash
python generate_data_from_seaward.py
# ex) python  generate_data_from_seaward.py --data_id 3
```

## Execution
```bash
python execute.py
# ex) python execute.py --data_path data --data_id 1 --neighborhood_size 100 --score_threshold 2.95
```

## Acknowledgements
The codes and datasets in this repository are based on the papers listed below. Thanks to the authors of these papers.
```
@inproceedings{alexiou2020towards,
  title={Towards a point cloud structural similarity metric},
  author={Alexiou, Evangelos and Ebrahimi, Touradj},
  booktitle={2020 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}

@article{krasnosky2022bathymetric,
  title={A bathymetric mapping and SLAM dataset with high-precision ground truth for marine robotics},
  author={Krasnosky, Kristopher and Roman, Christopher and Casagrande, David},
  journal={The International Journal of Robotics Research},
  volume={41},
  number={1},
  pages={12--19},
  year={2022},
  publisher={SAGE Publications Sage UK: London, England}
}

@inproceedings{tan2023data,
  title={Data-driven loop closure detection in bathymetric point clouds for underwater slam},
  author={Tan, Jiarui and Torroba, Ignacio and Xie, Yiping and Folkesson, John},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={3131--3137},
  year={2023},
  organization={IEEE}
}
```