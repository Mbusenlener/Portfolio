# Music Transformer 
This transformer-based music generator is trained autoregressively on a dataset of 51000 musical WAV format stereo-audio sequences and simple text pairings denoting genre and sentiment that I programmatically scraped from the web. Audio sequences are sampled as 30-second clips for training to reduce the computational burden of a large attentional context size stemming from a high sampling rate. To further combat costs in compute and memory, this model utilizes FAVOR+ attention and a Reversible architecture.

## Code Map
* compress.py: Preprocessing script for compressing training audio sequences with Encodec and exporting dataset as .h5 files for loading during training.
* model.py: Music Generator Transformer architecture implemented as a Pytorch module.
* deploy.py: Training deployment script that loads data files prepared by compress.py and initiates training.
* train.py: Training module used in deploy.py. Contains the core training loop.
* generate.py: Script for generating unique musical sequences given a pretrained model checkpoint produced from deploy.py. Takes an input description from the user, generates corresponding musical sequence, and exports the 10s sequence to mp3.

## Training
During training, this model learns to predict each note in a musical sequence given the notes that come before using causal masking. During preprocessing, all 30-second WAV audio sequences are encoded with Encodec, a pretrained high-fidelity neural audio codec, into in a discrete latent representation with four codebook features by 4500 timesteps and a token vocabulary size of 1024. This model predicts the four codebooks in parallel, leveraging a unique embedding layer for each parallel stream that projects the features onto a shared embedding space as input to the transformer. The outputs of the transformer are then projected with four unique dense layers onto the the audio token vocabulary space to produce logits for each of the four codebooks of the predicted timestep. During transformer decoding, this model performs causal masked self-attention on the embedded musical sequences as well as unmasked cross-attention on the text-description labels that have been encoded with a pretrained BERT language encoder. Through this training setup, the model learns to predict training sequences with text conditioning.

## Generation
After training, this model can generate unique musical sequences by taking a text description and unique designated start of sequence token as inputs. At each generative step, the previously generated sequence is fed into the model, and the next note's probability distribution across the token vocabulary space for each codebook is sampled with topk sampling. After a designated sequence length is reached, the generated sequence is decompressed using Encodec to produce a WAV stereo audio sequence that is saved as an MP3 file.

## Results
I trained this model for 20k steps with a batch size of 200 using AdamW optimizer. Training was deployed on one L4 GPU in a GCP VM instance. The model did not fully converge as I stopped training early since this was just a personal project using GCP free credits. However, even with a limited amount of training, my music transformer learned to produce relatively musical sounding sequences, despite some noticeable artifacts and noise. Below are four examples with corresponding text inputs that this trained model produced.
#### Chill Lofi: 


https://github.com/user-attachments/assets/d6873e7c-47ec-4a87-895d-fc32ef69912d
#### Upbeat Music:

https://github.com/user-attachments/assets/3774d84f-6984-4dc4-9a89-e31f571a941e
#### Baroque: 

https://github.com/user-attachments/assets/1c21fd01-e72e-4f06-a851-7c290275b293
#### Classical:

https://github.com/user-attachments/assets/6d243b2d-efc2-48d3-99cb-6e436e243c42

## Citations: 
  ```bibtex
  @misc{performer_pytorch,
    author = {Phil Wang},
    title = {Performer - Pytorch},
    year = {2022},
    url = {https://github.com/lucidrains/performer-pytorch},
  }
  @article{defossez2022highfi,
    title={High Fidelity Neural Audio Compression},
    author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
    journal={arXiv preprint arXiv:2210.13438},
    year={2022}
 }
  @inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
 }



