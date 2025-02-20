```py
#
#
#
#
import torch
import einops
import transformers


#
#
#
#
class Attention(torch.nn.Module):
  def __init__(self, dim, hds, isc=False):
    super().__init__()
    self.dim = dim # emb dimension
    self.hds = hds # number of heads
    self.isc = isc # is causal
    self.qkv_prj = torch.nn.Linear(dim, 3 * dim)
    self.out_prj = torch.nn.Linear(dim, dim)
    self.dropout = torch.nn.Dropout(0.1)

  def forward(self, x):
    _, seq, dim = x.size()
    hds = dim // self.hds
    qry, key, val = self.qkv_prj(x).split(dim, dim=-1)
    qry = einops.rearrange(qry, 'b s (h d) -> b h s d', h=self.hds, d=hds)
    key = einops.rearrange(key, 'b s (h d) -> b h s d', h=self.hds, d=hds)
    val = einops.rearrange(val, 'b s (h d) -> b h s d', h=self.hds, d=hds)
    att = torch.matmul(qry, key.transpose(-2, -1)) * (hds ** -0.5)
    if self.isc:
      msk = torch.triu(torch.ones((seq, seq), dtype=torch.bool, device=x.device), diagonal=1).view(1, 1, seq, seq)
      att = att.masked_fill(msk, float('-inf'))
    att = torch.nn.functional.softmax(att, dim=-1)
    att = self.dropout(att)
    out = torch.matmul(att, val)
    out = einops.rearrange(out, 'b h s d -> b s (h d)')
    out = self.out_prj(out)
    return out


#
#
#
#
class FFN(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.one = torch.nn.Linear(dim, dim)
    self.drp = torch.nn.Dropout(0.1)
    self.rlu = torch.nn.ReLU(inplace=True)
    self.two = torch.nn.Linear(dim, dim)

  def forward(self, x):
    x = self.one(x)
    x = self.rlu(x)
    x = self.drp(x)
    x = self.two(x)
    return x


#
#
#
#
class DecoderLayerSelf(torch.nn.Module):
  def __init__(self, dim, hds):
    super().__init__()
    self.c_attn = Attention(dim, hds, isc=True)
    self.ffn = FFN(dim)
    self.pre = torch.nn.LayerNorm(dim)
    self.ini = torch.nn.LayerNorm(dim)
    self.mid = torch.nn.LayerNorm(dim)
    self.fin = torch.nn.LayerNorm(dim)

  def forward(self, hdn):
    hdn = self.pre(hdn)
    out = self.c_attn(hdn)
    out = hdn + out
    out = self.ini(out)
    out = self.mid(out)
    fot = self.ffn(out)
    out = out + fot
    out = self.fin(out)
    return out


#
#
#
#
class Frank(torch.nn.Module):
  def __init__(self):
    super().__init__()
    clp_mode = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    self.enc = clp_mode.vision_model
    self.emb = clp_mode.text_model.embeddings
    self.cls = torch.nn.Linear(768, 512)
    self.dec = torch.nn.ModuleList([DecoderLayerSelf(512, 8) for _ in range(12)])
    self.voc = torch.nn.Linear(512, 49408, bias=False)
    with torch.no_grad(): self.voc.weight.data.copy_(self.emb.token_embedding.weight.data)
    self.enc.requires_grad_(False) # freeze
    self.emb.requires_grad_(False) # freeze

  def forward(self, img, ipt):
    img = self.enc(img)
    img = self.cls(img.pooler_output)
    img = img.unsqueeze(1)
    emb = self.emb(ipt)
    seq = torch.cat([img, emb], dim=1)
    for dec in self.dec: seq = dec(seq)
    seq = self.voc(seq)
    return seq

  def predict(self, img, pro):
    out = self.forward(img, pro)
    prd = torch.argmax(out, dim=-1)
    prd = torch.multinomial(torch.softmax(out[:, -1, :] / 0.8, dim=-1), 1)
    return torch.cat([pro, prd], dim=1)


#
#
#
#
if __name__ == '__main__':
  frk = Frank()
  x = torch.randn(1, 3, 224, 224)
  z = torch.randint(0, 49408, (1, 10))
  out = frk(x, z)
  print('out', out.shape)
  pass
```