# EAVT
Eco-Acoustic Visualization Tools

![plot](image.png)

## Audio processing
**First download the ResNet22 pretrained model using instructions [here](https://github.com/qiuqiangkong/audioset_tagging_cnn#audio-tagging-using-pretrained-models)**

process.py [-h] [--data_path DATA_PATH] [--save_path SAVE_PATH]

Script to process sound files recorded by Audiomoth

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to wav files
  --save_path SAVE_PATH
                        Path to save meta data

## Dash app

dash_app.py [-h] [--save_path SAVE_PATH]

Script to display sound files recorded by Audiomoth

optional arguments:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        Path to save meta data



Code for Audioset Tagging CNN from [Qiu Qiang Kong](https://github.com/qiuqiangkong/audioset_tagging_cnn)