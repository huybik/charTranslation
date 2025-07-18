# Character-level machine translation Transformer sequence to sequence in Pytorch
**Translate text with Transformer sequence to sequence architecture using only characters, implement in Pytorch**

<p align = "center"> 
<img src='https://github.com/huybik/charTranslation/blob/8a367afd51743efa856368cd43482a966c466d86/images/floating-man.jpg' width=400>
</p>
<p align = "center"> Light weight text translation </p>

Most implementations of text translation use word or byte-pair with 1000+ vocabularies as input which takes ages to train. This implementation use only characters in text data  (<300 chars) as vocabulary, so I can train everything from scratch in under 30 minutes on a GTX 1060. In this implementation you will find:
- Transformer and self attention re-implementation in Pytorch in ```encode_decode_transformer.py```
- A compact fast and greedy beam search implementation at the end of model
- Boilerplate for training ```trainer.py```
- Jupyter notebook to run model step by step

Sample translation results
```
thời tiết hôm nay thật đẹp! | <sos>the weather is concerned! it's beautiful!<eos>
xin chào | <sos>please come along<eos>
bạn đã ăn sáng chưa? | <sos>have you eaten the morning, didn't you?<eos>
```
This implementation is inspired by <a href='https://github.com/karpathy/minGPT'>Andrej Karpathy MinGPT</a>


