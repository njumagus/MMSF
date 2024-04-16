# MMSF: A Multimodal Sentiment-Fused Method to Recognize Video Speaking Style

All the data, processed data, extracted features, pretrained model weights, results, supporting sources and our docker image can be downloaded at [Baidu Disk](https://pan.baidu.com/s/1capDCX6_55jdSW8Yx3eGBg?pwd=4ac6) or [OneDrive](https://1drv.ms/f/s!AqIJSYD5gt-YlC-ecPHw0h9u2Dve). 

## 1.Quick Start
⚡ If you use the docker image that we provided and download all the data, features and model weights directly, you can go straight to perform inference.
```
sudo docker load -i yfraquelle_mmsf_env_v1.tar
sudo docker run -it -v /host/path/to/MMSF:/docker/path/to/MMSF --ipc=host --net=host <image_id> /bin/bash
cd /docker/path/to/MMSF
source activate mmsf
export PYTHONPATH=$PWD
python prediction/run_mmsf.py --test
```

## 2.Environment
- python3.8
- CUDA11.8
```
export PYTHONPATH=$PWD
```


## 3.Preparing Dataset
⚡ Considering that the raw videos were downloaded from YouTube but some of them are not publicly accessible now, we will not provide the raw data but provide the downloading script with videos' YouTube ids.
### Data Download
Download [LVU dataset](https://github.com/chaoyuaw/lvu) and unzip the data to data/LVU/raw.

Install python packages:
```
pip install pandas
pip install yt-dlp
```
Download videos:
```
python data_process/data_download.py
```
Now, the expected data structure is:
```
data/
  LVU/
    raw/
      director/
      genre/
      like_ratio/
      relationship/
      scene/
      view_count/
      way_speaking/
      writer/
      year/
    test/
      videos/
      test_dict.json
    train/
      videos/
      train_dict.json
    val/
      videos/
      val_dict.json
    data_dict.json
```

### Data Process
Install python packages:
```
pip install ffmpeg-python
pip install deepspeech
pip install scipy
```
Run scripts:
```
python data_process/video_process.py
python data_process/audio_process.py
python data_process/subtitle_process.py
```


## 4.Feature Extract
⚡ All the extracted data are compressed to "MMSA_feat.zip" in our Baidu Disk/OneDrive.

Install MMSA-FET following https://github.com/thuiar/MMSA-FET/wiki/Dependency-Installation: 
Download python packages:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install opencv-python
pip install python_speech_features
pip install scenedetect
pip install librosa
pip install opensmile
pip install transformers
pip install mediapipe
pip install gdown
pip install easydict
pip install pynvml
```
Install MSA_FET:
```
cd feat_extract
python -m MSA_FET install
cd ..
```

Download "bert-base-uncased" from the official link or the Baidu Disk/OneDrive that we provided and unzip it to the root. 
Download the feature extraction model weights from "feat_extract.zip" in the Baidu Disk that we provided and unzip it into the "feat_extract" folder. 
Download the "sources.zip" and unzip it to the root. 

Run scripts:
```
python feat_extract/MMSA_feat.py --label
python feat_extract/MMSA_feat.py --audio
python feat_extract/MMSA_feat.py --video
python feat_extract/MMSA_feat.py --text
python feat_extract/MMSA_senti_feat.py
```


## 5.Training and Inference
⚡ The pretrained model weights and results can be downloaded directly from "prediction.zip" in our Baidu Disk/OneDrive.

Install python packages:
```
pip install torchnet
```
Run scripts:
```
python prediction/run_mmsf.py --train
python prediction/run_mmsf.py --test
```

## 6.Citation
Please cite our paper if you find our work useful for your research:
```
@inproceedings{zhang2023mmsf,
  title={MMSF: A multimodal sentiment-fused method to recognize video speaking style},
  author={Zhang, Beibei and Fang, Yaqun and Yu, Fan and Bei, Jia and Ren, Tongwei},
  booktitle={ACM International Conference on Multimedia Retrieval},
  pages={289--297},
  year={2023}
}
```
