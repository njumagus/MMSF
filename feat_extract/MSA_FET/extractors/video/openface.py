import os
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd

from ..baseExtractor import baseExtractor


class openfaceExtractor(baseExtractor):
    """
    Video feature extractor using OpenFace. 
    Ref: https://mediapipe.dev/
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing OpenFace video feature extractor...")
            super().__init__(config, logger)
            if self.config['average_over'] < 1:
                self.pool_size = 1
                logger.warning("'average_over' is less than 1, set to 1.")
            else:
                self.pool_size = self.config['average_over']

            self.args = self._parse_args(self.config['args'])
            self.tool_dir = Path(__file__).parent.parent.parent / "exts" / "OpenFace"
            if platform.system() == 'Windows':
                self.tool = "./FeatureExtraction.exe" #self.tool_dir / "FeatureExtraction.exe"
            elif platform.system() == 'Linux':
                self.tool = "./FeatureExtraction" #self.tool_dir / "FeatureExtraction"
            else:
                raise RuntimeError("Cannot Determine OS type.")

            print("looking for OpenFace in", self.tool)

            #if not self.tool.is_file():
            #    raise FileNotFoundError("OpenFace tool not found.")
        except Exception as e:
            self.logger.error("Failed to initialize mediapipeExtractor.")
            raise e

    def _parse_args(self, args):
        res = []
        if 'hogalign' in args and args['hogalign']:
            res.append('-hogalign')
        if 'simalign' in args and args['simalign']:
            res.append('-simalign')
        if 'nobadaligned' in args and args['nobadaligned']:
            res.append('-nobadaligned')
        if 'tracked' in args and args['tracked']:
            res.append('-tracked')
        if 'pdmparams' in args and args['pdmparams']:
            res.append('-pdmparams')
        if 'landmark_2D' in args and args['landmark_2D']:
            res.append('-2Dfp')
        if 'landmark_3D' in args and args['landmark_3D']:
            res.append('-3Dfp')
        if 'head_pose' in args and args['head_pose']:
            res.append('-pose')
        if 'action_units' in args and args['action_units']:
            res.append('-aus')
        if 'gaze' in args and args['gaze']:
            res.append('-gaze')
        return res
    
    def extract(self, img_dir, video_name=None, tool_output=False):
        """
        Function:
            Extract features from video file using OpenFace.

        Parameters:
            img_dir: path to directory of images.
            video_name: video name used to save annotation images.
            tool_output: if False, disable stdout of OpenFace tool.

        Returns:
            video_result: extracted video features in numpy array.
        """
        try:
            args = self.args.copy()
            args.extend(['-fdir', str(Path(img_dir).absolute()), '-out_dir', str(Path(img_dir).absolute())])
            # if not tool_output:
            #     args.append('-quiet')
            cmd = str(self.tool) + " " + " ".join(args)
            print('here',cmd)
            t1 = time.perf_counter()
            prev = os.getcwd()
            os.chdir(self.tool_dir)
            os.system(cmd)
            os.chdir(prev)
            t2 = time.perf_counter()
            print('here cmd end/min:',((t2-t1)/60.0))

            name = Path(img_dir).stem
            df = pd.read_csv(Path(img_dir) / (str(name) + '.csv'))
            features, local_features = [], []
            for i in range(len(df)):
                local_features.append(np.array(df.loc[i][df.columns[5:]]))
                if (i + 1) % self.pool_size == 0:
                    features.append(np.array(local_features).mean(axis=0))
                    local_features = []
            if len(local_features) != 0:
                features.append(np.array(local_features).mean(axis=0))
            return np.array(features)
        except Exception as e:
            self.logger.error(f"Failed to extract video features with OpenFace from {video_name}.")
            raise e
