# EAVT
Eco-Acoustic Visualization Tools

# Currently in development


## How to use

First, create a directory with wav audio files named YYMMDD_HHMMSS. 
Then, use process.py to extract the eco-acoustic cues and the tagging.
Finally, use the dash_app.py script to run the web application.



![plot](image.png)

## Audio processing
**First download the ResNet22 pretrained model using instructions [here](https://github.com/qiuqiangkong/audioset_tagging_cnn#audio-tagging-using-pretrained-models)**

```
process.py [-h] [--data_path DATA_PATH] [--save_path SAVE_PATH]

Script to process sound files recorded by Audiomoth

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to wav files
  --save_path SAVE_PATH
                        Path to save meta data
  --name NAME           name of measurement
  --process_tagging PROCESS_TAGGING
                        Process tagging 0 or 1
  --process_indices PROCESS_INDICES
                        Process indices 0 or 1
  --Fmin FMIN           Freq min (filter)
  --Fmax FMAX           Freq max (filter)

```

Eco-acoustic indices can be added in the script indicies.py.
The name of the indice must be added in the list `name_indicies` and the calculation part in the function `compute_ecoacoustics`. 

Each wav file is divided into 10 second segments and converted to flac.


## Dash app

dash_app.py [-h] [--save_path SAVE_PATH]

Script to display sound files recorded by Audiomoth

options:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        Path to save meta data
  --name NAME           name of measurement




Code for Audioset Tagging CNN from [Qiu Qiang Kong](https://github.com/qiuqiangkong/audioset_tagging_cnn)
Code for Eco-acoustic indices from [Patric Guyot](https://github.com/patriceguyot/Acoustic_Indices) and [Sylvain Haupert and Juan Sebastian Ulloa](https://github.com/scikit-maad/scikit-maad)