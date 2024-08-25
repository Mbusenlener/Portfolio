import torch 
from model import MuseNet
from transformers import BertTokenizer
import numpy as np
from encodec import EncodecModel
import soundfile as sf
import torch.nn.functional as F
import h5py
from scipy.signal import butter, filtfilt
import librosa
import librosa.effects as effects
import torch.amp
from tqdm import tqdm

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output

def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token
#Function for autoregressively generating unique musical sequences from a unique start of sequence token by sampling the next note until 10 seconds of music are generated.
def generate(src,src_mask,num,max_seq_length, model,device, temperature=1.):
    start_note = torch.full((num,4,1),1024)  # shape of (B, K, seq_len)
    generated = start_note  # shape of (B, 4, 1)
    generated = generated.to(device)
    for i in tqdm(range(1500)):
        logits = model(src, generated,src_padding_mask=src_mask) # shape of (B, 4, seq_len, 1024)
        logits = logits.permute(0, 1, 3, 2)  # [B, K, 1024, seq_len]
        logits = logits[..., -1]  # [B x K x 1024]
        probs = F.softmax(logits/temperature, dim=-1)
        next_token = sample_top_k(probs,150)
        generated = torch.cat((generated, next_token), dim=-1)
    return generated[:,:,1:] #shape of (B, K, seq_len)

#Format generated latent sequence and normalization scales for Encodec decompression
def format(data,scales):
    data = data.view(data.size(0),data.size(1),-1,150)
    data = torch.permute(data,(2,0,1,3))
    new_data = [(codes,scale) for codes, scale in zip(data,scales[:data.size(0)])]
    return new_data

#Functions for cleaning generated audio sequences
def low_pass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
def enhance_harmonics(audio, fs):
    harmonic, percussive = librosa.effects.hpss(audio)
    return harmonic * 1.5 + percussive * 0.5
def time_stretch(audio, rate=1.0):
    return librosa.effects.time_stretch(audio, rate=rate)
def pitch_correct(audio, fs, n_steps=0):
    # Apply pitch correction; n_steps can be positive (up) or negative (down)
    return librosa.effects.pitch_shift(audio, sr=fs, n_steps=n_steps)
def clean_audio(audio, fs=48000):
    audio = np.transpose(audio,axes=(1,0))
    cleaned_audio = low_pass_filter(audio, cutoff=8000, fs=fs)
    cleaned_audio = enhance_harmonics(cleaned_audio, fs)
    cleaned_audio = time_stretch(cleaned_audio, rate=0.85)
    cleaned_audio = pitch_correct(cleaned_audio, fs, n_steps=0)
    cleaned_audio = np.transpose(cleaned_audio,axes=(1,0))
    return cleaned_audio

#Decompress and save batch of generated musical sequences
def decompress_data(encoded_audio,model,filename,clean=False):
    decoded_audio = model.decode(encoded_audio)
    decoded_audio = decoded_audio.permute(0, 2, 1).detach().cpu().numpy() #shape of (batch_size, seq_len, num_channels)
    for i, audio in enumerate(decoded_audio):
        if clean:
            audio = clean_audio(audio, fs=48000)
        sf.write(f"{filename}_{i+1}.mp3", audio, 48000, format='MP3')


def save(song,run):
    scales = []
    with h5py.File('data/scales.h5', 'r') as f:
        scales = f['dataset'][:]
    scales = torch.tensor(scales,dtype=torch.float,device=song.device)
    scales = scales.unsqueeze(1).unsqueeze(1)
    data = format(song,scales)
    model = EncodecModel.encodec_model_48khz().to(song.device)
    model.set_target_bandwidth(6.0)
    mp3_filename = f"song{run}"
    decompress_data(data,model,mp3_filename)


#Program for generating unique musical sequence with pretrained model checkpoint based on input text description.
if __name__ == "__main__":
    run = 1
    num = 1
    texts = []
    for i in range(num):
        text = input(f"Describe song #{i+1} that you would like to generate: ")
        texts.append(text)
    max_seq_length = 4500
    model_file = "checkpoints/checkpoint.pt"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length)
    src = encoded_inputs['input_ids'] #shape of (num,seq_length)
    src_mask = encoded_inputs['attention_mask']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = MuseNet(max_seq_length=max_seq_length).to(device)
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    model.load_state_dict(torch.load(model_file)['state_dict'])
    with torch.no_grad():
        song = generate(src,src_mask,num,max_seq_length,model,device) #shape of (1,4500)
        save(song,run)