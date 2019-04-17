# import paiv_utils as paiv
import argparse as ap
import subprocess
import pdb
import sys
import re
import os
import shutil
import glob
#shutil.rmtree('/folder_name')
from shutil import copyfile

# For Spectro gram creation
from scipy.io import wavfile
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pylab import *
import numpy as np

# For Pytorch Inference
import dl_utils as du
import preprocess_utils as pu



def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))

## function to process the inference data and write it to the video
#annotate_video(video_in, inference_file, output_file)



class SmartFormatterMixin(ap.HelpFormatter):
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return ap.HelpFormatter._split_lines(self, text, width)


class CustomFormatter(ap.RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


def _parser():
    parser = ap.ArgumentParser(description='Tool to annotate a video using PowerAI Vision '
                                           'Requires :'
                                           'Requires :'
                                           'Requires :'
                                           '  python score_exported_dataset.py --validate_mode=classification --model_url=https://129.40.2.225/powerai-vision/api/dlapis/8f80467f-470c-47f3-bf3c-ab7e0880a66b --data_directory=/data/work/osa/2018-10-PSEG/datasets_local/dv_97_classification_augmented_dataset-test',
                               formatter_class=CustomFormatter)

    parser.add_argument(
        '--input_video', action='store', nargs='?', required=False,
        help='S|--input_video=<video file name>'
             'Default: %(default)s')

    #parser.add_argument(
    #    '--model_url', action='store', nargs='?',
    #    required=True,
    #    help='S|--model_url=<deployed model endpoint>')
#
    #parser.add_argument(
    #    '--output_directory', action='store', nargs='?',
    #    required=True,
    #    help='S|--data_directory=<location of exported PAIV dataset>')
#
    #parser.add_argument(
    #    '--force_refresh', action='store', nargs='?', required=False,
    #    choices=[True,False], default=True,
    #    help='S|--force_refresh=[True|False] '
    #         'Default: %(default)s')

    parser.add_argument("-s", "--split-size",
                        dest = "splitsize",
                        help = "Split or chunk size in seconds, for example 10",
                        action = "store",
                        required=False
                        )

    args = parser.parse_args()

    return args



def main():
    # Parse command line argument


    args = _parser()
    # --model_url='https://129.40.2.225/powerai-vision/api/dlapis/bda90858-45e4-4ca6-8161-7d63436bb6c6' --input_video="/data/work/osa/2018-10-PSEG/Videos/transmission\ tower\ detection\ demo.mp4" --output_directory="/data/work/osa/2018-10-PSEG/Videos/temp"

    #def edit_video(input_video, model_url,output_directory, output_fn, overlay_fn, max_frames=50, force_refresh=True):

    #print(args)
    #args.force_refresh = True
    # Force inputs for now ...
    #args.input_video = "LukeClips.mp4"
    args.input_video = "vader_slice.mp4"
    args.splitsize   = 1
    output_directory = "./dvlk/"
    spectrogram_dir = "/gpfs/gpfs_gl4_16mb/s4s004/vanstee/2019-03-dlspec/" + output_directory

    output_file = "annotated_video.mp4"
    model = "StarWars_Luke_Darth_Background_dv_ep13_acc90.pt"
    
    for argk in vars(args) :
        nprint("{} -> {}".format(argk,vars(args)[argk]))

    # Cut the video into a bunch of spectrogram pngs ...
    pu.preprocess_video(args.input_video, output_directory, args.splitsize)
    
    # This step runs inference, and returns a dictionary of spectrogram_png to class
    annotations_dict = du.infer_spectrograms(model, spectrogram_dir, batch_size=16)
    
    # Write the movie
    du.annotate_video(args.input_video , args.splitsize, annotations_dict, output_directory, output_file)


if __name__== "__main__":
  main()
