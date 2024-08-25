from encodec import EncodecModel
from encodec.utils import convert_audio
import soundfile as sf
from urllib.request import urlopen
import torch
import numpy as np
import io
import json
import h5py


#Encodes each frame of audio into discrete latent representation
def encode_audio(frames,scales,model,device):
    with torch.no_grad():
        encoded_audio = []
        for frame, scale in zip(frames,scales):
            mini_batches = torch.split(frame,500)
            code_list = []
            for mini_batch in mini_batches:
                if mini_batch.size(0) == 500:
                    codes, _ = model._encode_frame(mini_batch.to(device))
                    code_list.append(codes)
            codes = torch.stack(code_list,dim=0)
            codes = codes.view(-1,codes.size(2),codes.size(3))
            encoded_frame = (codes,scale.to(device))
            encoded_audio.append(encoded_frame)
        return encoded_audio
    
#Formats encoded_audio for dataset
def format(encoded_audio):
    tracks = [encoded[0] for encoded in encoded_audio[:-1]]
    formatted = torch.cat(tracks,dim=-1).squeeze().cpu().numpy() #shape of (num_samples,codebooks,seq_length)
    scales = torch.tensor([encoded[1] for encoded in encoded_audio[:-1]]) #shape of (30,1,1)
    return formatted, scales

#Normalizes volume and saves scales for each second-long frame of audio
def normalize(x):
    channels, length = 2, 1426208
    segment_length = 48000
    stride = int((1-0.01)*segment_length) #47520
    scales = []
    frames = []
    print("starting normalization")
    for offset in range(0,length,stride):            
        frame = []
        for y in x:
            segment = y[:, offset: offset + segment_length]
            if segment.size(1) < segment_length:
            # Zero padding the tensor to have a size of 48000
                padding = segment_length - segment.size(1)
                segment = torch.nn.functional.pad(segment, (0, padding), 'constant', 0)
            frame.append(segment)
        frame = torch.stack(frame, dim=0)
        mono = frame.mean(dim=1,keepdim=True)
        volume = mono.pow(2).mean(dim=(2),keepdim=True).sqrt()
        scale = 1e-8 + volume
        frame = frame / scale
        scale = scale.mean(dim=0,keepdim=True)
        scale = scale.view(-1,1)
        scales.append(scale)
        frames.append(frame)
    return frames, scales

#Reads audio from url, converts to 48kHz sampling rate, normalizes volume, and encodes audio
def compress(tracks,model,sr,device):
    waveform_list = []
    labels = []
    for i, track in enumerate(tracks):
        url = track['url']
        label = track['label']
        labels.append(label.encode('utf-8','ignore'))
        waveform, sr = sf.read(io.BytesIO(urlopen(url).read()))
        waveform = torch.tensor(waveform,dtype=torch.float)
        waveform = waveform.transpose(1,0)
        waveform = convert_audio(waveform, sr, model.sample_rate, model.channels)
        waveform_list.append(waveform)
    frames, scales = normalize(waveform_list)
    encoded_audio = encode_audio(frames,scales,model,device)
    return encoded_audio, labels

#Preprocessing program step for converting each 30 second sample of music into discrete latent representations
#consisting of 4 codebooks by 4500 timesteps.
if __name__ == "__main__":
    tracks = []
    with open(f'labeled_tracks.json') as json_file:
        tracks = json.load(json_file)
    sublists = [tracks[i:i + 1000] for i in range(0, len(tracks), 1000)]
    sr = 44100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = EncodecModel.encodec_model_48khz().to(device)
    model.normalize = False
    model.set_target_bandwidth(6.0)
    scale_list = []
    label_list = []
    samples = []
    max_len = 0
    for i, track_list in enumerate(sublists[:-1]):
        print(f"starting batch {i+1}")
        encoded_audio, labels = compress(track_list,model,sr,device)
        trials, scales = format(encoded_audio)
        scale_list.append(scales)
        samples.append(trials)
        label_list.extend(labels)
        max_label_length = max(len(label) for label in labels)
        max_len = max(max_label_length,max_len)
    samples = np.asarray(samples).reshape((-1,4,4500))
    with h5py.File(f'data/training_data.h5', 'w') as f:
        f.create_dataset('dataset', data=samples, dtype=int)
        dt = h5py.string_dtype(encoding='utf-8', length=max_len)
        f.create_dataset('labels', data=label_list, dtype=dt)
    scale_list = torch.cat([scale.unsqueeze(0) for scale in scale_list])
    scale_list = scale_list.mean(dim=0)
    with h5py.File(f'data/scales.h5', 'w') as f:
        f.create_dataset('dataset',data=scale_list)
