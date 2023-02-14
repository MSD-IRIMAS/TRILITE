#!/bin/bash


file_names=('Adiac' 'ArrowHead' 'Beef' 'BeetleFly' 'BirdChicken' 'Car' 'CBF' 'ChlorineConcentration' 'CinCECGTorso' 'Coffee' 'Computers' 'CricketX' 'CricketY' 'CricketZ' 'DiatomSizeReduction' 'DistalPhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'Earthquakes' 'ECG200' 'ECG5000' 'ECGFiveDays' 'ElectricDevices' 'FaceAll' 'FaceFour' 'FacesUCR' 'FiftyWords' 'Fish' 'FordA' 'FordB' 'GunPoint' 'Ham' 'HandOutlines' 'Haptics' 'Herring' 'InlineSkate' 'InsectWingbeatSound' 'ItalyPowerDemand' 'LargeKitchenAppliances' 'Lightning2' 'Lightning7' 'Mallat' 'Meat' 'MedicalImages' 'MiddlePhalanxOutlineAgeGroup' 'MiddlePhalanxOutlineCorrect' 'MiddlePhalanxTW' 'MoteStrain' 'NonInvasiveFetalECGThorax1' 'NonInvasiveFetalECGThorax2' 'OliveOil' 'OSULeaf' 'PhalangesOutlinesCorrect' 'Phoneme' 'Plane' 'ProximalPhalanxOutlineAgeGroup' 'ProximalPhalanxOutlineCorrect' 'ProximalPhalanxTW' 'RefrigerationDevices' 'ScreenType' 'ShapeletSim' 'ShapesAll' 'SmallKitchenAppliances' 'SonyAIBORobotSurface1' 'SonyAIBORobotSurface2' 'StarLightCurves' 'Strawberry' 'SwedishLeaf' 'Symbols' 'SyntheticControl' 'ToeSegmentation1' 'ToeSegmentation2' 'Trace' 'TwoLeadECG' 'TwoPatterns' 'UWaveGestureLibraryAll' 'UWaveGestureLibraryX' 'UWaveGestureLibraryY' 'UWaveGestureLibraryZ' 'Wafer' 'Wine' 'WordSynonyms' 'Worms' 'WormsTwoClass' 'Yoga')
for file_name in "${file_names[@]}"; do
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate tfgpu
	python3 -u main.py -e fcn -d $file_name > logg_files/$file_name.txt
done 