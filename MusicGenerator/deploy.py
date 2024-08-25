import torch
import h5py
from train import train 
from transformers import BertTokenizer

def deploy():
    num_epochs = 100
    lr = .001
    max_seq_length = 4500
    data = []
    labels = []
    #Training data consisting of 51000 30 second encoded mp3 music clips with text labels indicating genre, sentiment, etc.
    file_path = 'data/training_data.h5'
    with h5py.File(file_path, 'r') as f:
        data = f['dataset'][:]
        labels = f['labels'][:]
    print(len(labels))
    print("Data loaded successfully.")
    labels = [label.decode('utf-8') for label in labels]
    #Tokenize training labels
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(labels, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length)
    src = encoded_inputs['input_ids']
    src_mask = encoded_inputs['attention_mask']
    tgt = torch.tensor(data,dtype=int)
    dataset = torch.utils.data.TensorDataset(tgt,src,src_mask)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size = 8, shuffle=True,num_workers=8,pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training beginning.")
    checkpoint_path = None #For resuming from prior checkpoint
    train(data_loader,num_epochs,lr,device,checkpoint_path=checkpoint_path)
    print("Training complete.")

if __name__ == "__main__":
    deploy()
