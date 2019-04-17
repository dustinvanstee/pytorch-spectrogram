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
#if(os.path.exists(data_dir)) 
from shutil import copyfile

# For Spectro gram creation
from scipy.io import wavfile
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pylab import *
import numpy as np


# For data processing !
import preprocess_utils as pu


def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))

def runcmd(mycmd) :
    cmdary = re.split("\s+", mycmd)
    nprint(cmdary)
    process = subprocess.Popen(cmdary, stdout=subprocess.PIPE)
    stdout, err = process.communicate()
    # print(stdout)
    return stdout



#
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


# Directory Structure
#(powerai-1.6.0) vanstee@p10a114:~/2019-03-dlspec/data/StarWars $ ls
#01_rawVideo            03_labeledSplitVideos  05_train  06_infer            README.md
#02_consolidatedVideos  04_labeledSplitPngs    05_val    old_remove_someday

def create_training_data(args, delete_data=True, train_val_split=(0.7,0.3)) :
    input_video_dir = args.input_video_dir 
    splitsize       = args.splitsize
    splitsize_dir   =  "1000ms"  
    
    run_preprocess_02_03 = True
    run_copy_04          = True
    run_split_05         = True

    if(splitsize != 1) :
        nprint("Error add logic for splitsize not equal to 1...")
   
    classes = glob.glob(input_video_dir + "02_consolidatedVideos/*")
    # just grab the leaf directory as class label ..
    classes = [x.split('/')[-1] for x in classes]
    
    # This part takes all the videos // splits them 
    for label in classes :
        # create output directory ...
        lsv_dir = input_video_dir + "/03_labeledSplitVideos/" + splitsize_dir + "/" + label
        lsp_dir = input_video_dir + "/04_labeledSplitPngs/" + splitsize_dir + "/" + label

        if(not(os.path.exists(lsv_dir))) :
            runcmd("mkdir -p {}".format(lsv_dir))
        if(not(os.path.exists(lsp_dir))) :
            runcmd("mkdir -p {}".format(lsp_dir))

        # grab all the files in .... 02_consolidatedVideos
        files_dir = input_video_dir + "/02_consolidatedVideos/" + label
        files = glob.glob(files_dir + "/*")
        for filename in files :
            if(run_preprocess_02_03) :
                pu.preprocess_video(filename, lsv_dir, splitsize, delete_data=delete_data)
    
    # Now cp from 03 to 04
    if(run_copy_04 == True) :
        for label in classes :
            lsv_dir = input_video_dir + "/03_labeledSplitVideos/" + splitsize_dir + "/" + label
            lsp_dir = input_video_dir + "/04_labeledSplitPngs/" + splitsize_dir + "/" + label
           
            glob_png = lsv_dir + "/*.png"
            files_png =  glob.glob( glob_png)
            for filename in files_png :
                runcmd("cp {} {}".format(filename,lsp_dir))

    # Now Random sample 04
    if(run_split_05 == True) :
        for label in classes :
            lsp_dir = input_video_dir + "/04_labeledSplitPngs/" + splitsize_dir + "/" + label
            train_dir = input_video_dir + "/05_train/" + splitsize_dir + "/" + label
            val_dir = input_video_dir + "/05_val/" + splitsize_dir + "/" + label
            if(not(os.path.exists(train_dir))) :
                runcmd("mkdir -p {}".format(train_dir))
            if(not(os.path.exists(val_dir))) :
                runcmd("mkdir -p {}".format(val_dir))
         
            glob_png = lsp_dir + "/*.png"
            files_png =  glob.glob( glob_png)
            nf = len(files_png)


            train_files_idx = np.random.choice(list(range(nf)), size=int(train_val_split[0]*nf), replace=False, p=None)
            train_files_idx = sorted(train_files_idx)
            for idx in range(nf) :
                #nprint("{} {}".format(idx,train_files_idx[0]))
                send_to_train = False
                if(len(train_files_idx) > 0) :
                    if(idx == train_files_idx[0]) :
                        #move file to train
                        send_to_train = True

                if(send_to_train == True) :
                    train_files_idx.pop(0)
                    cmd = "file {} {} move to train".format(idx,files_png[idx])
                    cmd = "cp {} {}".format(files_png[idx],train_dir)
                    

                else :
                    #move file to val
                    cmd = "file {} {} move to validation".format(idx,files_png[idx])
                    cmd = "cp {} {}".format(files_png[idx],val_dir)
                #nprint(cmd)
                runcmd(cmd)


def main():
    # Parse command line argument


    args = _parser()
    # --model_url='https://129.40.2.225/powerai-vision/api/dlapis/bda90858-45e4-4ca6-8161-7d63436bb6c6' --input_video="/data/work/osa/2018-10-PSEG/Videos/transmission\ tower\ detection\ demo.mp4" --output_directory="/data/work/osa/2018-10-PSEG/Videos/temp"

    args.input_video_dir = "/gpfs/gpfs_gl4_16mb/s4s004/vanstee/2019-03-dlspec/data/StarWars/"
    args.splitsize   = 1

    for argk in vars(args) :
        nprint("{} -> {}".format(argk,vars(args)[argk]))

    # Cut the video into a bunch of spectrogram pngs ...
    create_training_data(args)
    
 

if __name__== "__main__":
  main()
