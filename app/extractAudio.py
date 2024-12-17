#ONLY RUN FOR NEW VIDEOS IN AVLip folder (either in 0_real or 1_fake)
from moviepy.editor import VideoFileClip
import os

inputdir = 'eng/1_fake/'
outputdir ='eng/wav/1_fake/'

# Create output directory if it doesn't exist
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

# Loop through each file in the input directory
for filename in os.listdir(inputdir):
    if filename.endswith(".mp4"):
        actual_filename = filename[:-4]
        filepath = os.path.join(inputdir, filename)
        outfile = os.path.join(outputdir, f"{actual_filename}.wav")

        # Load the video file
        video = VideoFileClip(filepath)
        # Extract the audio and write it to the output file
        video.audio.write_audiofile(outfile, codec='pcm_s16le')
