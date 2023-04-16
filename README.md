# TransformerMultiDimTimeForecast
 
Make a Transformer for time series forecasting with PyTorch

It is very very simple as there are only two modules.

I can't say that if I am right, just post it for beginner.

Comments have Chinese, sorry for others.

## Reference

https://github.com/KasperGroesLudvigsen/influenza_transformer

https://github.com/fengjun321/Transformer_count

## Network Structure

Only two modules: 

1. Use `nn.Linear` to convert `[*, enc_feature_size] -> [*, d_model]`, `[*, dec_feature_size] -> [*, d_model]`, same in the opposite direction.

2. Use `nn.Transformer` in `output = model(src, tgt)`

    where `src: (S, E), tgt: (T, E), output: (T, E)`
    
    where `S` is the source sequence length, `T` is the target sequence length, `E` is the feature number

Overall view in Network Structure:

```python
def __init__(self,...):
    super(TransformerTS, self).__init__()
    self.transform = nn.Transformer(
        ...
    )
    self.pos = PositionalEncoding(d_model)
    self.enc_input_fc = nn.Linear(enc_feature_size, d_model)
    self.dec_input_fc = nn.Linear(dec_feature_size, d_model)
    self.out_fc = nn.Linear(d_model, dec_feature_size)

def forward(self, enc_input, dec_input):
    # embed_encoder_input: [enc_seq_len, enc_feature_size] -> [enc_seq_len, d_model]
    embed_encoder_input = self.pos(self.enc_input_fc(enc_input))

    # embed_decoder_input: [dec_seq_len, dec_feature_size] -> [dec_seq_len, d_model]
    embed_decoder_input = self.dec_input_fc(dec_input)

    # x: [dec_seq_len, d_model]
    x = self.transform(embed_encoder_input, embed_decoder_input)

    # x: [dec_seq_len, d_model] -> [dec_seq_len, dec_feature_size]
    x = self.out_fc(x)

    return x
```

## Usage

### Environment

Just PyCharm is ok. Other way to create venv is simple.

### Simple Verification

1. Run `Transformer\train_transformer.py` to train model.

2. Run `Transformer/test_transformer.py` to test model and get a plot showing prediction and label.

### Paras to Modify

1. Paras in `TransformerTS`, such as `enc_feature_size, dec_feature_size, d_model, nhead`

2. Data length such as `enc_seq_len, dec_seq_len`

3. Strategy such as DataLoader, optimizer, criterion

### Way to Modify

1. Modify dataloader.

    My dataloader is just get line from csv, and I haven't made batch. You may have precise need.
    
2. Add modules and care for dimensions.
