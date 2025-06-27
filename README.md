Automatic Audio tagging of large collection of soundscapes using PANNs - specific branch for BumbleBuzz project. 

## How to use

First, create a directory with wav audio files named YYMMDD_HHMMSS.wav/flac or What_EVER_YYMMDD_HHMMSS.wav/flac. 
Then, use process.py to perform buzz detection (+many other audio tagging).
Finally, use the dash_app.py script to run the web application.

![plot](image.png)

## Audio processing

```
python process.py [-h] [--data_path DATA_PATH] [--save_path SAVE_PATH] [--name NAME] [--audio_format AUDIO_FORMAT] [--l LENGTH_AUDIO_SEGMENT] [--save_audio_flac SAVE_AUDIO_FLAC] 

Script to process sound files recorded by Audiomoth

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to wav files
  --save_path SAVE_PATH
                        Path to save meta data
  --name NAME           name of measurement  
  --audio_format AUDIO_FORMAT
                        wav or flac
  --l LENGTH_AUDIO_SEGMENT
                        Window length in seconds, must be larger than 5
  --save_audio_flac SAVE_AUDIO_FLAC
                        Saving audio in flac format (needed to run visualization tool)

```
### Example

```
python3 process.py --save_path example/metadata/ --data_path example/metadata/audio_0002/ --name 0004 --audio_format flac --l 5 
```
Tagging of all flac files in folder example/metadata/audio_002 in non-overlapping chunks of 5 seconds. 

Each chunk is converted to flac and saved.

## Dash app

```
python dash_app.py [-h] [--save_path SAVE_PATH]

Script to display sound files recorded by Audiomoth

options:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        Path to save meta data
  --name NAME           name of measurement
```

Code for Audioset Tagging CNN from [Qiu Qiang Kong](https://github.com/qiuqiangkong/audioset_tagging_cnn)
