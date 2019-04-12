def runcmd(mycmd) :
    cmdary = re.split("\s+", mycmd)
    nprint(cmdary)
    process = subprocess.Popen(cmdary, stdout=subprocess.PIPE)
    stdout, err = process.communicate()
    # print(stdout)
    return stdout



# Takes an input video, and slices it up into png files of 1 sec
def preprocess_video(input_video, output_directory, args) :
    # only takes in mp4 for now ....
    chop_video_extract_audio(input_video, output_directory, args.splitsize)
    # This needs to be threaded !!!
    create_spectrograms(output_directory)

# writes large mp4 to 1 second mp4
def chop_video_extract_audio(input_video, output_directory="tmp", splitsize=1) :
    owd = os.getcwd()

    input_video_ary = input_video.split("/")
    file_name = input_video_ary[-1]
    path = input_video_ary[0:-2]

    #pdb.set_trace()

    # create output dir if it doesnt exist
    if(os.path.exists(output_directory)) :
        nprint("Removing files -> {}".format(output_directory))
        shutil.rmtree(output_directory)
    
    nprint("Creating output directory {}".format(output_directory))
    os.mkdir(output_directory)


    input_video_abs     = os.path.abspath(os.path.dirname(input_video)) + "/" + file_name
    input_video_abs_new = os.path.abspath(os.path.dirname(input_video)) + "/" + output_directory + "/"+ file_name

    nprint("Copy file {} to {}".format(input_video_abs, input_video_abs_new))
    copyfile(input_video_abs, input_video_abs_new)

  
    nprint("original working directory = {}".format(owd))
    mycmd = "python  {} -f {} -s {}".format(VIDEO_SPLITTER, input_video_abs_new,splitsize)
    nprint("mycmd = {}".format(mycmd))

    #pdb.set_trace()
    a = runcmd(mycmd)

    os.chdir(output_directory)
    mp4_list = glob.glob("*-*-*-*.mp4")
    for mp4 in mp4_list :
        mycmd =  "ffmpeg -i {} {}".format(mp4, mp4.replace("mp4", "wav"))
        a = runcmd(mycmd)
    os.chdir(owd)

    #convert_mp4_wav(indir,outdir)


## Converts wav file to spectrogram png file  
# This function is a bit expensive !!
def create_spectrograms(wav_dir) :

    os.chdir(wav_dir)
    wav_list = glob.glob( '*.wav')
    nprint("Processing {} wav files".format(len(wav_list)))
    for wav in wav_list :
        # Create the Spectrogram data structure
        samplingFrequency, signalData = wavfile.read(wav)
        if signalData.shape[-1] == len(signalData):
            channels = 1
        else:
            channels = signalData.shape[1]
        frequencies, times, spectrogram = signal.spectrogram(signalData, samplingFrequency)    


        # Save as a png ....
        #fig = plt.figure()
        #specify channel number in second parameter below as in signalData[:,0] means channel 0
        plt.specgram(signalData[:,0],Fs=samplingFrequency)
        plt.axis('tight')
        plt.axis('off')
        #plt.show() 
        png= wav.replace('wav','png')  
        plt.savefig(png, dpi = 100, bbox_inches='tight')
