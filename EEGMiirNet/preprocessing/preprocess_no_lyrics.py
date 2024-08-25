import mne 
import numpy as np
import librosa
import h5py
import logging
log = logging.getLogger(__name__)


STIMULUS_IDS = [11, 12, 13, 14, 21, 22, 23, 24]

#function from deepthought
def interpolate_bad_channels(raw): 
    if 'EXG5' in raw.info.ch_names and 'EXG6' in raw.info.ch_names:
        raw.drop_channels(['EXG5','EXG6'])
    if len(raw.info['bads']) > 0:
        raw.set_montage("biosemi64")
        raw.interpolate_bads()
    else: 
        print('No bad channels that need to be interpolated.')

def getTrialLen(stim_id,subjectID):
    if subjectID == '01' or subjectID == '04' or subjectID == '06' or subjectID == '07':
        if stim_id == 11:
            length = 13.8843
        elif stim_id == 12:
            length = 7.9123
        elif stim_id == 13:
            length = 8.9502
        elif stim_id == 14:
            length = 12.2381
        elif stim_id == 21:
            length = 8.3265
        elif stim_id == 22:
            length = 16.0162
        elif stim_id == 23:
            length = 9.2353
        else:
            length = 6.8709
    else:
        if stim_id == 11:
            length = 13.4645
        elif stim_id == 12:
            length = 7.7737
        elif stim_id == 13:
            length = 8.9502
        elif stim_id == 14:
            length =12.2381
        elif stim_id == 21:
            length = 8.3265
        elif stim_id == 22:
            length = 16.0021
        elif stim_id == 23:
            length = 9.2353
        else:
            length = 6.8709
    return length


def fix_events(raw, subjectID, stim_channel='STI 014'):
    events = mne.find_events(raw, stim_channel='STI 014')
    merged = list()
    for i, event in enumerate(events):
        id = event[2]
        #keep trials from condition 1 (represented by final digit) for songs 1-4 and 9-12 (encoded by ids {10-14,20-24,30-34,40-44} and {210-214,220-224,230-234,240-244})
        if id % 10 == 1 and id < 1000 and not(id >= 10 and id <= 44):
            if events[i+1][2] == 1000: # followed by audio onset represented by id 1000
                #onset = adjust_start_time(id,events[i+1][0],subjectID) #use audio onset as start
                merged.append([events[i+1][0], 0, id])
            else:
                #onset = adjust_start_time(id,event[0],subjectID)
                merged.append([event[0], 0, id])
    merged = np.asarray(merged, dtype=int)
    stim_id = raw.ch_names.index(stim_channel)
    raw._data[stim_id,:].fill(0)     # delete data in stim channel
    raw.add_events(merged)


subject_nums_list = ['01','04','06','07','09','11','12','13','14']
trials = []
labels_list = []
#read in data from raw files representing the different subjects. Each of the nine subjects has 40 trial recordings (5 iterations by 8 songs) that we are interested in. 
#We want the first 6.75 seconds of each recording, so the data shape for each trial is 64 electrodes by 3456 timestamps at 512 hz.
for w in subject_nums_list:
    fileName = "P{}-raw.fif".format(w)
    raw_ = mne.io.read_raw_fif(fileName, preload=True)
    interpolate_bad_channels(raw_)
    fix_events(raw_,w)
    eeg_picks = mne.pick_types(raw_.info, meg=False, eeg=True, eog=False, stim=False)
    raw_.filter(l_freq=0.5, h_freq=30, picks=eeg_picks,
            l_trans_bandwidth=0.1, h_trans_bandwidth=0.5, method='fft')
    events_ = mne.find_events(raw_, stim_channel='STI 014')
    #ica = read_ica("P{}-100p_64c-ica.fif".format(w))
    #raw_ = ica.apply(raw_,exclude=ica.exclude)
    stimuli = STIMULUS_IDS
    for stim_id in stimuli:
        trial_len = getTrialLen(stim_id,w)
        epoch_picks = mne.pick_types(raw_.info, meg=False,eeg=True,eog=False,stim=False,exclude=[])
        epochs = mne.Epochs(raw_,events=events_,event_id=(stim_id*10)+1,tmin=0,tmax=trial_len,proj=False,baseline=(0,0),picks=epoch_picks,preload=True)
        for trial in epochs.get_data(): 
            processed_trial = []
            for channel in trial:
                samples = channel
                samples -= samples.mean()
                s = samples[0:3456]
                s = librosa.util.normalize(s)
                processed_trial.append(s)
            trials.append(np.asarray(processed_trial,dtype=np.float32))
            labels_list.append(stim_id)

print(np.array(trials).shape)  #Shape of (360,64,3456) where each group of 40 trials representing one participant is composed of 8 groups of 5 representing each stimulus in order
print(np.array(labels_list).shape)  #Shape of (360,)
trials = np.array(trials).astype('float32')
for i, val in enumerate(labels_list):
    if (val == 11):
        val = 0
    elif (val == 12):
        val = 1
    elif (val == 13):
        val = 2
    elif (val == 14):
        val = 3
    elif (val == 21):
        val = 4
    elif (val == 22):
        val = 5
    elif (val == 23):
        val = 6
    elif (val == 24):
        val = 7
    labels_list[i] = val
labels_list = np.array(labels_list)
with h5py.File('data_no_lyrics.h5', 'w') as f:
    f.create_dataset('dataset', data=trials)
    f.create_dataset('labels', data=labels_list, dtype='i')













