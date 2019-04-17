import subprocess
import pdb
import sys
import re
import os
import shutil
import glob
from shutil import copyfile

# For Spectro gram creation
from scipy.io import wavfile
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pylab import *
import numpy as np

VIDEO_SPLITTER = "/gpfs/gpfs_gl4_16mb/s4s004/vanstee/2019-03-dlspec/video-splitter/ffmpeg-split.py"

def runcmd(mycmd) :
    cmdary = re.split("\s+", mycmd)
    nprint(cmdary)
    process = subprocess.Popen(cmdary, stdout=subprocess.PIPE)
    stdout, err = process.communicate()
    # print(stdout)
    return stdout

def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))


# Takes an input video, and slices it up into png files of 1 sec
def preprocess_video(input_video, output_directory, splitsize=1,  delete_data=True) :
    # only takes in mp4 for now ....
    chop_video_extract_audio(input_video, output_directory, splitsize, delete_data)
    # This needs to be threaded !!!
    # create_spectrograms_mp(output_directory)
    create_spectrograms_singlethreaded(output_directory)

# writes large mp4 to 1 second mp4
def chop_video_extract_audio(input_video, output_directory="./tmp", splitsize=1, delete_data=True) :
    owd = os.getcwd()

    input_video_ary = input_video.split("/")
    file_name = input_video_ary[-1]
    path = input_video_ary[0:-2]


    # create output dir if it doesnt exist
    if(os.path.exists(output_directory) and delete_data == True) :
        nprint("Removing files -> {}".format(output_directory))
        shutil.rmtree(output_directory)

    if(not(os.path.exists(output_directory))) :
        nprint("Creating output directory {}".format(output_directory))
        os.mkdir(output_directory)


    input_video_abs     = os.path.abspath(os.path.dirname(input_video)) + "/" + file_name
    input_video_abs_new = output_directory + "/"+ file_name

    nprint("Copy file {} to {}".format(input_video_abs, input_video_abs_new))
    copyfile(input_video_abs, input_video_abs_new)

  
    nprint("original working directory = {}".format(owd))

    nprint("---------------------------------------------------------------------------------------------")
    nprint("Splitting movie into {} second chunks.  This could take just a min or two ..... ".format(splitsize))
    nprint("---------------------------------------------------------------------------------------------")
    mycmd = "python  {} -f {} -s {} ".format(VIDEO_SPLITTER, input_video_abs_new,splitsize)
    nprint("mycmd = {}".format(mycmd))

    #pdb.set_trace()
    a = runcmd(mycmd)

    os.chdir(output_directory)
    mp4_list = glob.glob("*-*-*-*.mp4")
    for mp4 in mp4_list :
        mycmd =  "ffmpeg -loglevel panic -n -i {} {}".format(mp4, mp4.replace("mp4", "wav"))
        a = runcmd(mycmd)
    os.chdir(owd)

    #convert_mp4_wav(indir,outdir)


## Converts wav file to spectrogram png file  
# This function is a bit expensive !!
from queue import Queue
from threading import Thread

def create_spectrograms(wav, thread_id=99) :
    png= wav.replace('wav','png')
    matplotlib.use('Agg')
    if(not(os.path.isfile(png))) :
        nprint("Processing {} file in threaded mode. ThrID={}".format(wav,thread_id))
        samplingFrequency, signalData = wavfile.read(wav)
        if signalData.shape[-1] == len(signalData):
            channels = 1
        else:
            channels = signalData.shape[1]
        frequencies, times, spectrogram = signal.spectrogram(signalData, samplingFrequency)    
    
    
        # Save as a png ....
        #fig = plt.figure()
        #specify channel number in second parameter below as in signalData[:,0] means channel 0
        #pdb.set_trace()
        nprint("Creating Png file {}.  Stats = {} {}".format(png, len(signalData[:,0]), samplingFrequency) )
        plt.rcParams["figure.figsize"] = [12,8]
        plt.specgram(signalData[:,0],Fs=samplingFrequency)
        plt.axis('tight')
        plt.axis('off')
        #plt.show() 
          
        plt.savefig(png, dpi = 100, bbox_inches='tight')
        plt.close()
    else :
        nprint("Png file {} already exists.  Not creating".format(png))

def create_spectrograms_singlethreaded(wav_dir) :
    matplotlib.use('Agg')
    orig_dir = os.getcwd()
    os.chdir(wav_dir)
    wav_list = glob.glob( '*.wav')
    nprint("Processing {} wav files".format(len(wav_list)))
    for wav in wav_list :
        # Create the Spectrogram data structure

        create_spectrograms(wav)
    
    os.chdir(orig_dir)



def create_spectrograms_threaded(wav_dir) :
    num_threads = 160
    max_pngs  = 1000
    def consume_pngs(q, result_dict, thread_id):
        
        while (q.qsize() > 0):

            (png_name) = q.get()
            if(q.qsize() % 10 == 0) :
                print("Thr {} : Size of queue = {}".format(thread_id, q.qsize()))
                #print("Thr {} : Hash key = {}".format(thread_id, frame_key))
                #print("Thr {} : Frame id = {}".format(thread_id, frame_id))

            rv = create_spectrograms(png_name )
            result_dict[png_name] = rv
            q.task_done()

    q = Queue(maxsize=0)
    res_hash = {}

    orig_dir = os.getcwd()
    os.chdir(wav_dir)
    wav_list = glob.glob( '*.wav')
    nprint("Adding {} wav files to queue".format(len(wav_list)))
    png_cnt = 0
    for wav in wav_list :
        if(png_cnt < max_pngs) :
            q.put((wav))
            png_cnt += 1

   # Setup Consumers.  They will fetch frame json info from api, and stick it in results list
    nprint("Consuming all pngs in queue.  Numthreads = {}".format(num_threads))

    threads = [None] * num_threads
    for i in range(len(threads)):
        threads[i] = Thread(target=consume_pngs, args=(q, res_hash, i))
        threads[i].start()

    # Block until all consumers are finished ....
    nprint("Waiting for all consumers to complete ....")
    for i in range(len(threads)):
        threads[i].join()

import multiprocessing

def create_spectrograms_mp(wav_dir) :
    num_threads = 160
    max_pngs  = 1000

    orig_dir = os.getcwd()
    os.chdir(wav_dir)
    wav_list = glob.glob( '*.wav')
    nprint("Adding {} wav files to queue".format(len(wav_list)))
    png_cnt = 0
    p = [None] * len(wav_list)
    for wav in wav_list :
        if(png_cnt < max_pngs) :
            p[png_cnt] = multiprocessing.Process(target=create_spectrograms, args=(wav,png_cnt))
            p[png_cnt].start()
            png_cnt += 1

   # Setup Consumers.  They will fetch frame json info from api, and stick it in results list


    # Block until all consumers are finished ....
    nprint("Waiting for all consumers to complete ....")
    for i in range(len(wav_list)):
        p[i].join()


