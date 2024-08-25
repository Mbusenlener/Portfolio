import torch
import torch.nn as nn
import torch.optim as optim
from model import MuseNet

def train(data_loader, num_epochs, lr, device, max_grad_norm=1.0,checkpoint_path=None):
    model = MuseNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
    accum_iter=25
    while epoch < num_epochs:
        model.train()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        running_loss = 0.0
        for i, (tgt, src, src_mask) in enumerate(data_loader):
            tgt, src, src_mask = tgt.to(device), src.to(device), src_mask.to(device)
            sos = torch.full((tgt.size(0),4,1),1024).to(device) #Unique start of sequence token
            tgt = torch.cat((tgt,sos), dim=-1)
            tgt_output = tgt[:,:,1:] #Right shifted target for loss calculation
            if ((i)%accum_iter == 0):
                print('Forwarding batch')
            output = model(src,tgt[:,:,:-1],src_padding_mask=src_mask) #shape of (B,K,seq_len,vocab_size)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            loss = loss / accum_iter
            loss.backward()
            running_loss += loss.item()
            if ((i+1)%accum_iter == 0):
                print(f'Backpropagating batch')
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{(int)((i+1)/accum_iter)}/{(int)(len(data_loader)/accum_iter)}], Loss: {running_loss:.4f}')
                running_loss = 0.0
        end.record()
        torch.cuda.synchronize()
        print(f'Time taken for Epoch {epoch+1}: {start.elapsed_time(end)/1000} seconds')
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint,f"checkpoints/checkpoint.pt")
        epoch += 1
