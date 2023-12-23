# MMSF: A Multimodal Sentiment-Fused Method to Recognize Video Speaking Style

All the data, processed data, extracted features, pretrained model weights, results and supporting sources can be downloaded at [Baidu Disk](https://pan.baidu.com/s/1capDCX6_55jdSW8Yx3eGBg?pwd=4ac6).

## Environment
- python3.8
- CUDA11.8
```
export PYTHONPATH=$PWD
```
## Preparing Dataset
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

## Feature Extract
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
Run scripts:
```
python feat_extract/MMSA_feat.py --label
python feat_extract/MMSA_feat.py --audio
python feat_extract/MMSA_feat.py --video
python feat_extract/MMSA_feat.py --text
python feat_extract/MMSA_senti_feat.py
```
## Training
Install python packages:
```
pip install torchnet
```
Run scripts:
```
python prediction/run_mmsf.py --train
```
## Inference
Run scripts:
```
python prediction/run_mmsf.py --test
```
