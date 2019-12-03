import pyedflib
import numpy as np
import pandas as pd

import math

import scipy.signal as signal

from scipy.signal import hilbert

import arrow

from datetime import datetime as dt
from datetime import timedelta as td

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

def get_edf_signal_indices(f, signals):
    """Read EDF signals with the given names into a matrix"""
    labels = [l.lower() for l in f.getSignalLabels()]
    indices = [labels.index(signal.lower()) for signal in signals]
    return indices


def read_edf_signals(f, signals):
    """Read EDF signals with the given names into a matrix"""
    indices = get_edf_signal_indices(f, signals)
    sigbufs = np.zeros((len(signals), f.getNSamples()[indices[0]]))
    for i, signal_index in enumerate(indices):
        # print(f"read {i} from {signal_index}")
        sigbufs[i, :] = f.readSignal(signal_index)
    return sigbufs


def get_edf_frequencies(f, signals):
    indices = get_edf_signal_indices(f, signals)
    frequencies = []
    for label, i in zip(signals, indices):
        i_freq = int(f.samplefrequency(i))
        frequencies.append(i_freq)
    return frequencies


def read_edf_signals_file(filename, signals):
    """Read EDF signals and close file"""
    f = pyedflib.EdfReader(filename)
    sigs = read_edf_signals(f, signals)
    f._close()
    return sigs


def resample_signals(signals, orig_freq=200, new_freq=64):
    """Resample the given signals to the provided frequency"""
    resampled_signals = []
    for sig in signals:
        new_length = int(len(sig)*new_freq/orig_freq)
        resampled = signal.resample(sig, new_length)
        resampled_signals.append(resampled)

    return resampled_signals


def preprocess(abd, chest, orig_freq=200):
    '''
    sigs[0] should be ABD
    sigs[1] should be Chest
    '''
    # Resample to 64Hz
    abd_64, chest_64 = resample_signals([abd, chest], orig_freq)


    bands =   (0, 0.05, 0.1, 5, 10, 32)
    desired = (0, 0,    1,   1, 0, 0)
    b_firwin = signal.firwin2(73, bands, desired, fs=64)
    #b_firls = signal.firls(73, bands, desired, fs=64)

    abd_64_filt = signal.filtfilt(b_firwin, 1, abd_64)
    chest_64_filt = signal.filtfilt(b_firwin, 1, chest_64)
    return abd_64_filt, chest_64_filt


def fund_freq(s, freq=64, nfft=1024):
    f, Pxx_den = signal.welch(s, freq)
    ff = f[np.argmax(Pxx_den)]
    return ff


def avg_resp_period(abd, chest):
    """
    It was estimated based on the fundamental frequencies of the
    RC and ABD signals of the entire recording, by using the
    Welch power spectral density estimation method
    with 50% window overlap.
    """
    # Mean of fundamental frequency
    ff = np.mean(list(map(fund_freq, [abd, chest])))
    avg_resp_period = 1/ff
    # print(avg_resp_period)
    return avg_resp_period


def calc_windows(length, win_size, stride):
    return (length - win_size) // stride + 1

def test_calc_windows():
    assert calc_windows(21, 5, 2) == 9
    assert calc_windows(21, 7, 2) == 8
    assert calc_windows(21, 5, 3) == 6
    assert calc_windows(21, 5, 5) == 4
    assert calc_windows(20, 5, 5) == 4
    assert calc_windows(19, 5, 5) == 3

def phase_angle_method_2():
    """Not working"""
    x1_h = hilbert(x1)
    x2_h = hilbert(x2)
    x1_r = np.real(x1_h)
    x1_i = np.imag(x1_h)
    x2_r = np.real(x2_h)
    x2_i = np.imag(x2_h)

    numer = x1*x2_h - x2*x1_h
    denom = x1*x2 + x1_h*x2_h

    numer = x1_r*x2_i - x2_r*x1_i
    denom = x1_r*x2_r + x1_i*x2_i
    taa = np.arctan2(numer, denom)


def calc_taa(abd, chest, freq=64):
    """Calculate the time in asynchrony per second.

    Result size depends on the stride size based on the
    average respiratory period.
    """
    resp_period = avg_resp_period(abd, chest)
    # The length of the window was set to three times the average respiratory period.
    win_length_sec = resp_period*3
    # The step size of the sliding window was set to a quarter of the average respiratory period.
    step_size_sec = resp_period/4
    print(f"avg resp period {resp_period} win {win_length_sec} step size {step_size_sec}")

    x1 = abd
    x2 = chest
    win_size = int(win_length_sec*freq)
    stride = int(step_size_sec * freq)
    xlen = len(x1)
    win_count = calc_windows(xlen, win_size, stride)
    taa_raw = np.zeros(win_count)
    taa_valid = np.full(win_count, True, dtype=np.bool)
    taa_invalid_reason = np.zeros(win_count, dtype=np.uint32)
    for i in range(win_count):
        j = i * stride
        j_end = j + win_size
        if j_end > xlen:
          break

        x1b = x1[j:j_end]
        x2b = x2[j:j_end]

        # Check for validity
        # if 1024 is slow, try 512
        f1, Pxx_den1 = signal.welch(x1b, freq, nfft=1024)
        ff1 = f1[np.argmax(Pxx_den1)]
        f2, Pxx_den2 = signal.welch(x2b, freq, nfft=1024)
        ff2 = f2[np.argmax(Pxx_den2)]

        # ratio of spectral power within the frequency band of interest to total power < 0.65
        in_band_power_1 = np.sum(Pxx_den1[2:10])
        in_band_power_2 = np.sum(Pxx_den2[2:10])
        total_power_1 = np.sum(Pxx_den1)
        total_power_2 = np.sum(Pxx_den2)
        INVALID_ABD_LOW_POWER = 1
        INVALID_CHEST_LOW_POWER = 2
        INVALID_ABD_FREQ_OUT_OF_RANGE = 4
        INVALID_CHEST_FREQ_OUT_OF_RANGE = 8
        INVALID_CHEST_ABD_FREQ_DISPARITY = 16
        if in_band_power_1 == 0.0 or in_band_power_1 / total_power_1 < 0.65:
            taa_invalid_reason[i] |= INVALID_ABD_LOW_POWER
            taa_valid[i] = False
        if in_band_power_2 == 0.0 or in_band_power_2 / total_power_2 < 0.65:
            taa_invalid_reason[i] |= INVALID_CHEST_LOW_POWER
            taa_valid[i] = False

        # breathing frequencies lie outside the physiological range for children (0.12 - 0.585Hz)
        if ff1 < 0.12 or ff1 > 0.585:
            taa_invalid_reason[i] |= INVALID_ABD_FREQ_OUT_OF_RANGE
            taa_valid[i] = False
        if ff2 < 0.12 or ff2 > 0.585:
            taa_invalid_reason[i] |= INVALID_CHEST_FREQ_OUT_OF_RANGE
            taa_valid[i] = False

        # disparity between RC and ABD fundamental frequencies existed (defined by a difference > 20%)
        ff_larger = max(ff1, ff2)
        ff_smaller = min(ff1, ff2)
        if ff_larger != ff_smaller and (ff_larger-ff_smaller) / ff_larger  > 0.5:
            taa_invalid_reason[i] |= INVALID_CHEST_ABD_FREQ_DISPARITY
            taa_valid[i] = False


        x1_h = hilbert(x1b)
        x2_h = hilbert(x2b)
        c = np.inner(x1_h, np.conj(x2_h)) / np.sqrt(np.inner(x1_h, np.conj(x1_h)) * np.inner(x2_h,np.conj(x2_h)))
        phase_angle = np.abs(np.angle(c))
        #phase_angle = np.arctan(c)
        taa_raw[i] = phase_angle
    taa = np.abs(taa_raw)/np.pi
    return taa, taa_invalid_reason, 1/step_size_sec


def calc_sec_taa(abd, chest, win_size=3, freq=64, stride=1):
    """Calculate the time in asynchrony per second.

    Result is in seconds based on the given frequency.
    """
    x1 = abd
    x2 = chest
    # win_size = 3
    x_stride = win_size*freq
    half_x_stride = x_stride // 2
    xlen = len(x1)
    win_count = int(np.ceil(xlen/x_stride))
    secs = int(len(x1)/freq)
    taa = np.zeros(secs)
    for i in range(win_count):
        j = max(i * x_stride - half_x_stride, 0)
        j_end = min(i * x_stride + half_x_stride, xlen)
        x1_h = hilbert(x1[j:j_end])
        x2_h = hilbert(x2[j:j_end])
        c = np.inner(x1_h, np.conj(x2_h)) / np.sqrt(np.inner(x1_h, np.conj(x1_h)) * np.inner(x2_h,np.conj(x2_h)))
        phase_angle = np.abs(np.angle(c))
        #phase_angle = np.arctan(c)
        taa[i*win_size:i*win_size+win_size] = phase_angle
    taa_raw = taa.copy()
    taa = np.abs(taa)/np.pi
    return taa


def read_sleep_stages(total_duration, scored_events):
    """
    Read sleep stage information from the annotation file.

    Result is in seconds.


    """
    #print(len(scored_events))
    stages = np.zeros(total_duration, dtype=np.int8)
    for event in scored_events:
        event_type = event.eventtype.text.split('|')[0]
        if event_type == 'Stages':
            #print('type:', event_type)
            #print('concept:', event.eventconcept.text)
            #print('start:', event.start.text)
            #print('duration:', event.duration.text)
            concept = event.eventconcept.text.split('|')
            stage = concept[0]
            stage_no = int(concept[1])
            start = int(float(event.start.text))
            duration = int(float(event.duration.text))
            stages[start:start+duration] = stage_no
    # Convert any stage 4s to 3s
    stages = np.where(stages == 4, 3, stages)
    return stages


def read_resp_events(total_duration, scored_events):
    """
    Read sleep stage information from the annotation file.

    Result is in seconds.


    """
    #print(len(scored_events))
    resp_events = np.zeros(total_duration, dtype=np.int8)
    TYPE_CODE = {
        'Hypopnea': 1,
        'Obstructive apnea': 2,
        'Central apnea': 3,
        'SpO2 desaturation': 4,
        'SpO2 artifact': 5,
        'Unsure': 6,
        'Other': 7,
    }
    for event in scored_events:
        event_type = event.eventtype.text.split('|')[0]
        if event_type == 'Respiratory':
            #print('type:', event_type)
            #print('concept:', event.eventconcept.text)
            #print('start:', event.start.text)
            #print('duration:', event.duration.text)
            concept = event.eventconcept.text.split('|')
            resp_type = concept[0]
            type_code = TYPE_CODE.get(resp_type, 7)
            if type_code == 7:
                # print(f"Event type{concept}")
                pass
            start = int(float(event.start.text))
            duration = int(float(event.duration.text))
            resp_events[start:start+duration] = type_code
    return resp_events


def chart_taa(start_time,
              abd, chest,
              taa, taa_freq, taa_valid,
              stages, start_dt, window_len=30):
    plt.figure(figsize=(16,16))
    freq = 64
    #start_time = 5_600
    #start_time = 8_000
    #start_time = 11_474
    #start_time = 5_600

    # Change 5 to 4 to make the chart easier
    chart_stages = np.where(stages == 5, 4, stages)

    plot_range = np.arange(start_time*freq, (start_time+window_len)*freq)

    time_range = np.empty_like(plot_range, dtype='datetime64[ms]')
    for i in range(len(plot_range)):
        time_range[i] = start_dt.naive + td(milliseconds=plot_range[i]*1/freq*1000)

    # sleep stage is different resolution
    ss_time_range = np.empty(window_len, dtype='datetime64[ms]')
    for i in range(window_len):
        ss_time_range[i] = start_dt.naive + td(seconds=start_time+i)


    # taa time range
    taa_len = int(window_len*taa_freq)
    taa_time_range = np.empty(taa_len, dtype='datetime64[ms]')
    # TEMP TODO -12/taa_freq is a guess and matches the delay,
    # but not sure it's valid. VERIFY
    # I got 12 because the window length is 12 times the step size
    taa_start = int(start_time*taa_freq - 12/taa_freq)
    #taa_start = int(start_time*taa_freq - 6/taa_freq)
    taa_end = taa_start + int(window_len*taa_freq)
    for i in range(taa_len):
        taa_time_range[i] = start_dt.naive + td(seconds=start_time*taa_freq + i)


    plt.subplot(411)
    plt.title('Sleep Stage')
    plt.ylim((-0.2,4.2))
    plt.yticks([0, 1, 2, 3, 4], ['Awake 0', 'Sleep 1', 'Sleep 2', 'Sleep 3/4', 'REM Sleep 5'])
    plt.plot(ss_time_range, chart_stages[start_time:start_time+window_len])

    plt.subplot(412)
    plt.title('ABD / Chest')
    plt.plot(time_range, abd[plot_range])
    #plt.subplot(413)
    #plt.title('Chest')
    plt.plot(time_range, chest[plot_range])
    plt.subplot(413)
    plt.title('TAA')
    plt.ylim(0,100)
    plt.plot(taa_time_range, taa[taa_start:taa_end]*100)
    plt.plot(taa_time_range, taa_valid[taa_start:taa_end], 'y')


def compute_perc_taa(filename='baseline/chat-baseline-300004',
                    base_path='local-data/chat/polysomnography',
                    ):
    ANNOT_DIR = 'annotations-events-nsrr'
    EDF_DIR = 'edfs'

    edf_file = f'{base_path}/{EDF_DIR}/{filename}.edf'
    annot_file = f'{base_path}/{ANNOT_DIR}/{filename}-nsrr.xml'

    f = pyedflib.EdfReader(edf_file)
    with open(annot_file) as fp:
        annot = BeautifulSoup(fp)

    # Sanity checking
    edf_duration = f.getFileDuration()
    annot_duration = int(float(annot.scoredevents.scoredevent.duration.text))
    assert edf_duration == annot_duration
    duration_sec = edf_duration

    signal_labels = ['ABD','Chest']
    edf_indices = get_edf_signal_indices(f, signal_labels)
    orig_freq = int(f.samplefrequency(edf_indices[0]))
    for label, i in zip(signal_labels, edf_indices):
        i_freq = int(f.samplefrequency(i))
        assert orig_freq == i_freq, "Abd and chest readings must have the same frequency"

    print(f"{filename} sample frequency is {orig_freq}.")

    sigs = read_edf_signals(f, signal_labels)

    f._close()

    # This will resample to 64Hz
    abd, chest = preprocess(sigs[0], sigs[1], orig_freq=orig_freq)

    stages = read_sleep_stages(annot_duration, annot.scoredevents.find_all('scoredevent'))
    resp_events = read_resp_events(annot_duration, annot.scoredevents.find_all('scoredevent'))

    sleep_stages = stages != 0
    awake_seconds = np.sum(stages == 0)
    sleep_seconds = np.sum(sleep_stages)
    stage_1_seconds = np.sum(stages == 1)
    stage_2_seconds = np.sum(stages == 2)
    stage_3_seconds = np.sum(stages == 3)
    stage_5_seconds = np.sum(stages == 5)

    # Filter sleeping taa also by resp events
    """
    events = [
        'Normal',
        'Hypopnea',
        'Obstructive apnea',
        'Central apnea',
        'SpO2 desaturation',
        'SpO2 artifact',
        'Unsure',
        'Other',
    ]"""
    # Filter 1, 2, 3 Hypopnea and both apneas
    no_resp_events_filter = (resp_events != 1) & (resp_events != 2) & (resp_events != 3)
    hyp_apnea_seconds = np.sum(~no_resp_events_filter)
    
    # Filter chest and abd signals before calculating taa
    # because the taa calculation is not in seconds
    
    # keep where sleep_stages is True and no_resp_events_filter is True
    # and expand boolean to match frequency of abd and chest
    asleep_filter = np.repeat(sleep_stages, 64)
    asleep_no_events_filter = np.repeat(sleep_stages & no_resp_events_filter, 64)
    asleep_non_event_seconds = np.sum(sleep_stages & no_resp_events_filter)
    
    # taa = calc_taa(abd, chest)
    asleep_non_event_taa, taa_valid, taa_freq = calc_taa(abd[asleep_no_events_filter], chest[asleep_no_events_filter])
    asleep_taa, taa_valid_events, taa_freq_events = calc_taa(abd[asleep_filter], chest[asleep_filter])
    
    # Filter taa and resp events by time asleep
    # asleep_taa = taa[sleep_stages]
    # asleep_events = resp_events[sleep_stages]

    invalid_taa_time = np.sum(taa_valid != 0)

    # Sanity check the parsing
    # The taa isn't in seconds any more with the new calculation for stride length
    # print("duration vs taa:", duration_sec, len(taa))
    # assert duration_sec == len(taa)
    # assert duration_sec == len(stages)
    # assert duration_sec == len(resp_events)

    result = 'success'

    # Return a vector of values:
    # length of file
    # length of time sleeping
    # length of time in a resp event
    # time_in_async for various cutoffs?
    return (
        filename,
        result,
        orig_freq,
        duration_sec,
        # awake_seconds,
        sleep_seconds,
        # stage_1_seconds,
        # stage_2_seconds,
        # stage_3_seconds,
        # stage_5_seconds,
        hyp_apnea_seconds,
        asleep_non_event_seconds,
        np.round(np.sum(asleep_non_event_taa > 0.25) / len(asleep_non_event_taa), 4),
        np.round(np.sum(asleep_non_event_taa > 0.50) / len(asleep_non_event_taa), 4),
        np.round(np.sum(asleep_non_event_taa > 0.75) / len(asleep_non_event_taa), 4),
        np.round(np.sum(asleep_taa > 0.25) / len(asleep_taa), 4),
        np.round(np.sum(asleep_taa > 0.50) / len(asleep_taa), 4),
        np.round(np.sum(asleep_taa > 0.75) / len(asleep_taa), 4),
        invalid_taa_time,
    )


def read_files(files, base_path='local-data/chat/polysomnography'):
    dtypes = np.dtype([
        ('filename', np.object),
        ('comment', np.object),
        ('orig_freq', np.int64),
        ('duration_sec', np.int64),
        # ('awake_seconds', np.int64),
        ('sleep_seconds', np.int64),
        # ('stage_1_seconds', np.int64),
        # ('stage_2_seconds', np.int64),
        # ('stage_3_seconds', np.int64),
        # ('stage_5_seconds', np.int64),
        ('hyp_apnea_seconds', np.int64),
        ('asleep_non_event_seconds', np.int64),
        ('time_in_async_25p', np.float64),
        ('time_in_async_50p', np.float64),
        ('time_in_async_75p', np.float64),
        ('time_in_async_25p_wevents', np.float64),
        ('time_in_async_50p_wevents', np.float64),
        ('time_in_async_75p_wevents', np.float64),
        ('invalid_taa_periods', np.int64),
    ])
    df = np.empty(len(files), dtypes)
    for i, filename in enumerate(files):
        print(f"{i:3} processing {filename}")
        df[i] = compute_perc_taa(filename, base_path=base_path)
    results_df = pd.DataFrame(df, index=files)
    results_df = results_df.drop(columns='filename')
    results_df.index.name = 'filename'
    return results_df
