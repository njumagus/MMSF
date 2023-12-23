import os
import json
from data.config import pass_videos
import pandas as pd

def csv2json():
    class_dict = {}
    video_info = {}
    for mode in ['train','val','test']:
        if not os.path.exists('data/LVU/'+mode):
            os.mkdir('data/LVU/'+mode)
        video_info[mode] = {}
        mode_video_info = {}
        csv_path = 'data/LVU/raw/way_speaking/' + mode+'.csv'
        label_df = pd.read_csv(csv_path,dtype={'class_id class_name youtube_id imdb_id':str})
        for i in range(0, len(label_df)):
            item = label_df.iloc[i]
            info = item[item.keys()[0]]
            class_id = info.split(' ')[0]
            class_name = info.split(' ')[1]
            youtube_id = info.split(' ')[2]
            imdb_id = info.split(' ')[3]
            video_info[mode][str(i)] = {'video_id':i,'video_name':str(i).zfill(4)+'.mp4','class_id':class_id,'youtube_id':youtube_id,'imdb_id':imdb_id}
            mode_video_info[str(i)] = {'video_id':i,'video_name':str(i).zfill(4)+'.mp4','class_id':class_id}
            if class_id not in class_dict:
                class_dict[class_id] = class_name
        json.dump(mode_video_info, open('data/LVU/'+mode+'/'+mode+'_dict.json','w'))
    json.dump({'class_dict':class_dict, 'video_info':video_info}, open('data/LVU/data_dict.json', 'w'))

def download_data():
    for mode in ["train","val","test"]:
        video_info_dict = json.load(open('data/LVU/data_dict.json','r'))
        video_info = video_info_dict['video_info'][mode]

        for key in video_info:
            video_name = video_info[key]['video_name']
            video_id = video_info[key]['video_id']
            if video_id in pass_videos['lvu'][mode]:
                continue
            youtube_id = video_info[key]['youtube_id']
            video_path = 'data/LVU/'+mode+"/videos/"+video_name
            # en_subtitle_path = 'data/LVU/'+mode+"/subtitles/"+str(video_id).zfill(4)+'.en.vtt'
            # fi_subtitle_path = 'data/LVU/'+mode+"/subtitles/"+str(video_id).zfill(4)+'.fi.vtt'
            if not os.path.exists(video_path):
                video_download_cmd = "yt-dlp 'https://www.youtube.com/watch?v="+youtube_id+"' -f mp4 -o 'data/LVU/"+mode+"/videos/"+video_name+"'"
                print(video_download_cmd)
                os.system(video_download_cmd)

csv2json()
download_data()