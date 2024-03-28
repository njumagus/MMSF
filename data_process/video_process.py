#没法去水印，因为有的水印直接打在场景画面上了
#这个脚本是为了讲视频的后30s广告去掉
import ffmpeg
import os
import json
from data.config import pass_videos

class VideoProcess():
    def __init__(self,mode):
        self.mode=mode
        self.video_info_dict = json.load(open('data/LVU/data_dict.json','r'))
        self.video_info = self.video_info_dict['video_info'][mode]
        self.pass_videos = pass_videos['lvu'][mode]

        self.data_root = 'data/LVU/'+mode
        self.video_dir = self.data_root + '/videos'
        self.processed_videos_dir=os.path.join(self.data_root,'processed_videos')

        self.imgsize = 224
        self.npylen = 16

    # 这个函数是为了把movieclip视频后面三十秒的广告剪掉
    def video_cut(self):
        if not os.path.exists(self.processed_videos_dir):
            os.makedirs(self.processed_videos_dir)
        for key in self.video_info:
            video_name = self.video_info[key]['video_name']
            video_id = self.video_info[key]['video_id']
            if video_id in self.pass_videos:
                continue
            print(video_id)
            input_path = os.path.join(self.video_dir,video_name)
            if not os.path.isfile(input_path):
                continue
            output_path = os.path.join(self.processed_videos_dir,video_name)
            info = ffmpeg.probe(input_path)
            duration = int(float(info['streams'][0]['duration']))
            target_duration = duration - 30
            cmd = 'ffmpeg -ss 0 -i '+input_path+' -t '+str(target_duration)+' -c:v copy -c:a copy '+output_path
            # print(cmd)
            os.system(cmd)

for mode in ["train",'val','test']:
    video_process=VideoProcess(mode)
    video_process.video_cut()


