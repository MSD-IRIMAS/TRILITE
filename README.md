# Enhancing Time Series Classification with Self-Supervised Learning

This is the code of our paper "[Enhancing Time Series Classification with Self-Supervised Learning](https://doi.org/10.5220/0011611300003393)" [[pdf]](https://hal.science/hal-04143083/document) accepted as short paper at [2023 15th International Conference on Agents and Artificial Interlligence](https://icaart.scitevents.org/). <br> This work was done by [Ali Ismail-Fawaz](https://hadifawaz1999.github.io/), [Maxime Devanne](https://maxime-devanne.com/), [Jonathan Weber](https://www.jonathan-weber.eu/) and [Germain Forestier](https://germain-forestier.info/).

## Triplet Loss Mechanism

The Triplet Loss used in this work was proposed by [F. Schroff et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html).

<p align='center'>
<img style="height: auto; width: 400px" src='figures/model.png'/>
</p>

## Triplet Generation

### Positive Samples

<p align='center'>
<img style="height: auto; width: 500px" src='figures/pos.png'/>
</p>

### Negative Samples

<p align='center'>
<img style="height: auto; width: 500px" src='figures/neg.png'/>
</p>

## Dataset

This model has been tested on the 85 univariate time series datasets of the [UCRArchive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/)

## Usage of code

To run the code of the TRILITE model on a dataset of the UCRArchive, simply run ```python3 main.py -e fcn -d Coffee```

To run the code on the TRILITE model in a semi-supervised way on a dataset of the UCRArchive,<br /> simply run ```python3 main_semi_supervised.py -e fcn -d Coffee -p 30```, ```-p``` is to specify the percentage of the semi-supervised split.

To apply classication on the results, simply run ```python3 apply_classifier -e fcn -o results/ -d Coffee``` or <br /> 
```python3 apply_classifier -e fcn -o results_semi_30/ -d Coffee``` to apply classification on the semi-supervised split results.

To avoid running each time on one dataset at the time, simply use the bash files ```run_ucr.sh```, ```run_semi_ucr.sh``` and ```run_classifier_ucr.sh``` for each case.

### Adaptation of Code

The change that should be done is the directory in which the datasets are stored.
The variable to be changed is [this line](https://github.com/MSD-IRIMAS/TRILITE/blob/e860e5b657cf8daf127862fde781ec1de86f2fcb/utils/utils.py#L23) ```folder_path```.

## Results

Results of the conducted experiments can be found in the two csv files, one for the [enhancement of FCN](results/fcn/results_ucr.csv) and one for the [Semi-Supervised setup](results_semi_30/fcn/results_ucr.csv).

### TRILITE with a Fully Connected Layer Classifier

<p align='center'>
<img style="height: auto; width: 300px" src='figures/trilit_vs_fcn.png'/>
</p>

### TRILITE+FCN with a Fully Connected Layer Classifier (+ means concatenation of features)

<p align='center'>
<img style="height: auto; width: 300px" src='figures/concat_vs_fcn.png'/>
</p>

### TRILITE+FCN in a Semi-Supervised environment with a RIDGE Classifier

<p align='center'>
<img style="height: auto; width: 300px" src='figures/semi_30_compare.png'/>
</p>

### Latent Space Representation in 2D using TRILITE on the SyntheticControl Dataset of the UCR archive

<p align='center'>
<img style="height: auto; width: 400px" src='figures/tsne.png'/>
</p>

## Reference

If you use this code, please cite our paper:

```
@inproceedings{ismail-fawaz2022trilite,
  author = {Ismail-Fawaz, Ali and Devanne, Maxime and Weber, Jonathan and Forestier, Germain},
  title = {Enhancing Time Series Classification with Self-Supervised Learning},
  booktitle = {15th International Conference on Agents and Artificial Intelligence: ICAART 2023},
  city = {Lisbon},
  country = {Portugal},
  pages = {1--8},
  year = {2023},
  publisher={SciTePress},
  organization = {INSTICC}
}
```


## Acknowledgments

This work was supported by the ANR DELEGATION project (grant ANR-21-CE23-0014) of the French Agence Nationale de la Recherche. The authors would like to acknowledge the High Performance Computing Center of the University of Strasbourg for supporting this work by providing scientific support and access to computing resources. Part of the computing resources were funded by the Equipex Equip@Meso project (Programme Investissements d’Avenir) and the CPER Alsacalcul/Big Data. The authors would also like to thank the creators and providers of the UCR Archive.
