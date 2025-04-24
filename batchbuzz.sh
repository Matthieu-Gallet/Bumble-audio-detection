dirlist=$1

for curdir in $(cat $dirlist)
do
python process.py --save_audio_flac 1 --data_path /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/$curdir --name $curdir --Fmax 15999 --length_audio_segment 5 --process_tagging 1 --save_path /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/detection --audio_format wav
done