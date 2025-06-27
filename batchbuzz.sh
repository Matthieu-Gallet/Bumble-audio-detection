dirlist=$1

for curdir in $(cat $dirlist)
do
python process.py --save_audio_flac 1 --data_path /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/$curdir --name $curdir --l 5 --save_path /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/detection_Cnn10 --model_type Cnn10 --audio_format wav
done

rm -Rf /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/detection_Cnn10/audio_*

for curdir in $(cat $dirlist)
do
python process.py --save_audio_flac 1 --data_path /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/$curdir --name $curdir --l 5 --save_path /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/detection_Cnn14 --model_type Cnn14 --audio_format wav
done

rm -Rf /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/detection_Cnn14/audio_*

for curdir in $(cat $dirlist)
do
python process.py --save_audio_flac 1 --data_path /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/$curdir --name $curdir --l 5 --save_path /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/detection_wglmCnn14 --model_type Wavegram_Logmel_Cnn14 --audio_format wav
done

rm -Rf /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/detection_wglmCnn14/audio_*
