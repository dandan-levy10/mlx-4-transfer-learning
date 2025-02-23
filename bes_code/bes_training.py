#
#
#
#
import tqdm
import torch
import datetime
import bes_dataset as dataset
import bes_model as model
import wandb


#
#
#
#
torch.manual_seed(42)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


#
#
#
#
frk = model.Frank().to(dev)
torch.save(frk.state_dict(), f'./checkpoints/{ts}.0.frk.pth')
print('frk:enc', sum(p.numel() for p in frk.enc.parameters())) # 87,456,000
print('frk:emb', sum(p.numel() for p in frk.emb.parameters())) # 25,336,320
print('frk:voc', sum(p.numel() for p in frk.voc.parameters())) # 25,336,320
print('frk:dec', sum(p.numel() for p in frk.dec.parameters())) # 18,960,384
print('frk:tot', sum(p.numel() for p in frk.parameters()))     # 157,444,352


#
#
#
#
ds = dataset.Flick(split='train')
dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.collate)


#
#
#
#
opt = torch.optim.Adam(frk.parameters(), lr=0.0001)
crt = torch.nn.CrossEntropyLoss(ignore_index=ds.tk.pad_token_id)
wandb.init(project='mlx6-week-04-frk')


#
#
#
#
for epoch in range(5):
  pgs = tqdm.tqdm(dl, desc=f'Epoch {epoch + 1}', leave=False)
  for idx, (txt, img) in enumerate(pgs):
    img = img.to(dev)
    txt = txt.to(dev)
    out = frk(img, txt)
    out = out[:, 1:-1, :]
    lbl = txt[:, 1:]
    out = out.permute(0, 2, 1)
    opt.zero_grad()
    loss = crt(out, lbl)
    loss.backward()
    opt.step()
    wandb.log({'loss': loss.item()})
  torch.save(frk.state_dict(), f'./checkpoints/{ts}.{epoch + 1}.frk.pth')


#
#
#
#
wandb.finish()