# TRILITE: Triplet Loss In Time

## Dataset

This model has been tested on the 85 univariate time series datasets of the [UCRArchive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/)

## Running the code

To run the code of the TLIT model on a dataset of the UCRArchive, simply run ```python3 main.py -e fcn -d Coffee```

To run the code on the TLIT model in a semi-supervised way on a dataset of the UCRArchive,<br /> simply run ```python3 main_semi_supervised.py -e fcn -d Coffee -p 30```, ```-p``` is to specify the percentage of the semi-supervised split.

To apply classication on the results, simply run ```python3 apply_classifier -e fcn -o results/ -d Coffee``` or <br /> 
```python3 apply_classifier -e fcn -o results_semi_30/ -d Coffee``` to apply classification on the semi-supervised split results.

To avoid running each time on one dataset at the time, simply use the bash files for each case.
