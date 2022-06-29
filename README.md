# Licence Plate Detection

## Requirements
I would recommend creating a virtual enviroment
```
git clone https://github.com/sashanktalakola/licence-plate-detection.git
cd licence-plate-detection
virtualenv -p python3 venv
. ./venv/bin/activate
git clone https://github.com/alexeyab/darknet/ darknet
pip3 install streamlit opencv-python
```
Few changes need to be made to the Makefile<br>
If you have GPU and CUDA support (It helps running the application fast) run
```
cd darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
sed -i 's/LIBSO=0/LIBSO=1/' Makefile
```
Incase you don't
```
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/LIBSO=0/LIBSO=1/' Makefile
```
If ```sed -i 's/OPENCV=0/OPENCV=1/' Makefile``` is causing any error you don't need to run it (Set ```OPENCV=0``` in ```Makefile```)Just installing ```opencv-python``` should also work fine<br><br>

```
cd .. #Root directory of project
make
```
Before you run the application few changes need to be made<br>
Inside the ```darknet``` folder create an empty file ```__init__.py```
```
touch darknet/__init__.py
```
In the ```darknet.py``` file (Located at ```darknet/darknet.py```) change relative path to absolute path in lines ```236``` and ```240``` (This is just to prevent a few errors)<br>

Finally to run the application
```
streamlit run app.py
```

# Optional
## Download data
Create two new folders to download data and annotations (Annotations is actually optional it isn't necessary to run the application)
```
oi_download_dataset --base_dir <DOWNLOAD-DIR> --labels "Class 1" "Class 2" --format darknet --csv_dir <ANNOTATIONS-PATH> --limit <NUM-IMAGES>
```
Incase you don't need annotations file
```
oi_download_dataset --base_dir <DOWNLOAD-DIR> --labels "Class 1" "Class 2" --format darknet --limit <NUM-IMAGES>
```
You can download as many images you would like, but I would recommend ```2000*num_classes```. Here num_classes = 1 (Vehicle registration plate)
```
mkdir download annotations
oi_download_dataset --base_dir ./download --labels "Vehicle registration plate" --format darknet --csv_dir ./annotations --limit 2500
```
I will be using ```2000``` image for training and remaining ```500``` for validation purpose<br>
Then run ```split-data.py``` script to split images into training and testing folders or you could manually pick images for training and validation
```
python3 split-data.py
```


## Getting your data ready
Incase of multiclass detection update ```train.names``` and ```train.data``` files accordingly
Next we need to generate ```train.txt``` and ```test.txt``` files it can be done by running ```generate_train.py``` and ```generate_test.py``` files
```
python3 generate_train.py
python3 generate_test.py
```

