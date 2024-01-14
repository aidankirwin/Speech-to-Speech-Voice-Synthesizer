from scipy.io import wavfile
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import tkinter as tk
from tkinter import filedialog

# helper function for plotting data
def plot(x, y, title, x_label, y_label):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def filter(data, freq, bandwidth):
    # butterworth bandpass
    min_cutoff = freq - bandwidth / 2
    max_cutoff = freq + bandwidth / 2

    passband_range = [min_cutoff, max_cutoff]
    sos = butter(8, passband_range, btype='bandpass', output='sos', fs=int(SAMPLE_RATE))

    # filtered data
    filtered = sosfilt(sos, data)
    return np.abs(filtered)

def RMS(dataset):
    length = len(dataset)
    sum_of_squares = 0
    for x in dataset:
        sum_of_squares += x*x
    
    root = np.sqrt(sum_of_squares)

    return root / length

# uses signal.resample to set a dataset's sample rate to the const variable SAMPLE_RATE
def downSample(data, init_sr):
    num_of_samples = round(len(data) * (SAMPLE_RATE) / init_sr)
    f = signal.resample(data[:], num_of_samples)

    return f

def timeChunk(f, s, gap):
    chunks = []

    for i in range(CHUNK_COUNT):
        min = int(i * s)
        max = int(((i + 1) * s))

        if min != 0:
            min += gap

        if max > len(f) - 1:
            break

        temp = f[min:max]
        
        # create new Chunk object and add to chunks
        new_chunk = Chunk()
        new_chunk.time_chunk = temp
        new_chunk.start_time = min
        new_chunk.end_time = max

        chunks.append(new_chunk)

    return chunks

def synthChunk(chunks, s):
    min_freq = 50
    max_freq = 7000
    band_count = int((max_freq - min_freq) / BANDWIDTH) # number of freq bands
    audio_chunks = []

    for chunk in chunks:
        # get fft
        fft = rfft(chunk.time_chunk)
        chunk.fft = fft

        # initialize synthesized chunk array
        chunk.synth = [0] * int(s)

        # chunk time as linspace
        c_time = np.linspace(int(chunk.start_time), int(chunk.end_time), int(s)) / SAMPLE_RATE

        curr_freq = min_freq + BANDWIDTH / 2 # first centre frequency in loop
        for i in range(band_count):
            # filter and take RMS
            filtered_band = filter(chunk.time_chunk, curr_freq, BANDWIDTH)
            # add to Chunk class' RMS dictionary
            chunk.freq_RMS[curr_freq] = RMS(filtered_band)

            # add sin of current frequency and calculated RMS to Chunk object
            w = curr_freq * 2 * np.pi
            a = chunk.freq_RMS[curr_freq]
            new_synth = a * GAIN * np.sin(c_time * w)

            for j in range(len(chunk.synth)):
                chunk.synth[j] += new_synth[j]

            curr_freq += BANDWIDTH

        audio_chunks.append(chunk.synth)

    return audio_chunks

class Chunk:
    start_time = None
    end_time = None
    time_chunk = [] # chunk data in time domain
    fft = [] # chunk freq data in freq domain
    freq_RMS = {} # RMS for each freq band in freq domain
    synth = [] # synthesized chunk data in time domain

# ---------------- PREPROCESSING ----------------- #
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

rate, data = wavfile.read(file_path)
data = data.astype(float)

# convert from stereo to mono
if data.shape[1] == 2:
    data = data[:, 0]

# down sample to 16kHz
SAMPLE_RATE = 16_000 # hertz
f = downSample(data, rate)
# ------------- END OF PREPROCESSING -------------- #

# ---------------- TIME CHUNKING ----------------- #
chunks = [] # array of all chunks as Chunk objects
CHUNK_SIZE = 10 # milliseconds
s = CHUNK_SIZE * SAMPLE_RATE / 1000 # samples per chunk
CHUNK_COUNT = int(np.ceil(len(f) / s)) # number of chunks
GAP_SIZE = 0 # set to positive number for gaps, negative for overlap; milliseconds

chunks = timeChunk(f, s, GAP_SIZE)
# ------------- END OF TIME CHUNKING ------------- #

# ------------------ SYNTHESIS ------------------- #
GAIN = 200 # gain on synthesized signal
BANDWIDTH = 100 # hertz

synth_audio_chunks = synthChunk(chunks, s)
synth_audio = np.concatenate(synth_audio_chunks)
# --------------- END OF SYNTHESIS ---------------- #

synth_fft = rfft(synth_audio)
synth_xf = rfftfreq(len(synth_audio), 1 / SAMPLE_RATE)

signal_fft = rfft(f)
signal_xf = rfftfreq(len(f), 1 / SAMPLE_RATE)

time0 = np.linspace(0, 5, len(synth_audio))
time1 = np.linspace(0, 5, len(f))

plot(time1, f, "Original Audio", "Time (s)", "Magnitude")
plot(time0, synth_audio, "Synthesized Audio", "Time (s)", "Magnitude")
plot(signal_xf, 20*np.log10(np.abs(signal_fft)), "Original Audio", "Frequency (Hz)", "Magnitude")
plot(synth_xf, 20*np.log10(np.abs(synth_fft)), "Synthesized Audio", "Frequency (Hz)", "Magnitude")

f_int16 = synth_audio.astype(np.int16)
wavfile.write("example.wav", SAMPLE_RATE, f_int16)