import os
import json
import subprocess
from data.config import pass_videos

class AudioProcess():
    def __init__(self,mode):
        self.mode = mode
        self.pass_videos=pass_videos['lvu'][mode]
        self.data_root = os.path.join('data','LVU',mode)
        self.video_list = json.load(open(os.path.join(self.data_root,mode+'_dict.json'),'r'))
        self.videos_dir = os.path.join(self.data_root,'videos')
        self.wavs_dir = os.path.join(self.data_root,'aud_wav')
        if not os.path.exists(self.wavs_dir):
            os.makedirs(self.wavs_dir)


    def mp42wav(self):
        for key in self.video_list:
            video_name = self.video_list[key]['video_name']
            video_id = self.video_list[key]['video_id']
            if video_id in self.pass_videos:
                continue
            print(video_id)
            video_path = os.path.join(self.videos_dir,video_name)
            if not os.path.isfile(video_path):
                continue
            target_audio_path = os.path.join(self.wavs_dir,str(video_id).zfill(4)+'.wav')
            call_list = ['ffmpeg']
            call_list += ['-v', 'quiet']
            call_list += [
                '-i',
                video_path,
                '-ar',
                '16000',
                '-ac',
                '1',
                '-f',
                'wav']
            call_list += ['-map_chapters', '-1']  # remove meta stream
            call_list += [target_audio_path]
            subprocess.call(call_list)

for mode in ['train','val','test']:
    audio_process=AudioProcess(mode)
    audio_process.mp42wav()


