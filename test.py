import torch

model = MyModel(*args, **kwargs)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
ckp_path = "path/to/checkpoint/checkpoint.pt"
model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']