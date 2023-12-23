from deepspeech import Model
from scipy.io import wavfile
import os
import json
from data.config import pass_videos

class SubtitleProcess():
    def __init__(self,mode):
        self.mode = mode
        self.data_root = os.path.join('data','LVU',mode)
        self.video_info_dict = json.load(open(os.path.join('data','LVU','data_dict.json'),'r'))
        self.train_video_info = self.video_info_dict['video_info']['train']
        self.val_video_info = self.video_info_dict['video_info']['val'] #119和193是私人视频，看不了
        self.test_video_info = self.video_info_dict['video_info']['test']

        self.video_info = self.video_info_dict['video_info'][mode]
        self.pass_videos = pass_videos['lvu'][mode]
        self.wavs_dir = os.path.join(self.data_root,'aud_wav')
        self.subtitle_dir = os.path.join(self.data_root,'subtitles_txt')

    def speechtotext(self):
        if not os.path.exists(self.subtitle_dir):
            os.makedirs(self.subtitle_dir)
        model_path = 'sources/deepspeech/deepspeech-0.9.3-models.pbmm'
        ars = Model(model_path)
        for key in self.video_info:
            video_id = self.video_info[key]['video_id']
            if video_id in self.pass_videos:
                continue
            print(video_id)
            audio_path = os.path.join(self.wavs_dir,str(video_id).zfill(4)+'.wav')
            # audio_file = wave.open(audio_path, 'r')
            fs, data = wavfile.read(audio_path)
            translate_txt = ars.stt(data)
            with open(os.path.join(self.subtitle_dir,str(video_id).zfill(4)+'.txt'),'w') as f:
                f.write(translate_txt)

for mode in ['train','val','test']:
    subtitle_process=SubtitleProcess(mode)
    subtitle_process.speechtotext()



