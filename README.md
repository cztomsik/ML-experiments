My personal learning space. Nothing to see here, except maybe for the links below.

## Interesting Papers

- 2010 Understanding the difficulty of training deep feedforward neural networks\
  Original paper for xavier init\
  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

- 2015 All you need is a good init\
  LSUV init for conv nets.\
  https://arxiv.org/abs/1511.06422

- 2017 Attention is all you need\
  https://arxiv.org/abs/1706.03762

- 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\
  Encoder-only, SOTA NLP inference at the time.\
  https://arxiv.org/abs/1810.04805

- 2019 RoBERTa: A Robustly Optimized BERT Pretraining Approach\
  Improved BERT, Masking-only, dynamic masking, bigger batches, more data\
  New SOTA\
  https://arxiv.org/abs/1907.11692

- 2019 Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks\
  Sentence to vector. Vectors can be compared independently.\
  https://arxiv.org/abs/1908.10084

- 2019 T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer\
  https://arxiv.org/abs/1910.10683

- 2019 One epoch is all you need\
  https://arxiv.org/abs/1906.06669

- 2020 GPT-3: Language Models are Few-Shot Learners\
  Large enough model is capable of predicting the likely output from few examples in its prompt without any fine-tuning\
  https://arxiv.org/abs/2005.14165

- 2020 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\
  Transformers can be used for image recognition.\
  https://arxiv.org/abs/2010.11929

- 2020 Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention\
  Transformer rewritten as RNN can be used for speeding-up the inference.\
  https://arxiv.org/abs/2006.16236

- 2021 FLAN: Finetuned Language Models Are Zero-Shot Learners\
  https://arxiv.org/abs/2109.01652
  https://github.com/google-research/FLAN/blob/main/flan/templates.py

- 2021 An Attention Free Transformer\
  Alternative Transformer design\
  https://arxiv.org/abs/2105.14103
  - AFT-inspired time-mixing variation (WKV) with RNN/GPT mode, very fast\
    https://github.com/BlinkDL/RWKV-LM

- 2021 (ALiBi) Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation\
  Alternative position embedding\
  https://arxiv.org/abs/2108.12409

- 2022 FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\
  Very cool, but it needs custom CUDA kernel.\
  https://github.com/HazyResearch/flash-attention
  https://arxiv.org/abs/2205.14135

- 2022 Chinchilla\
  50B model can outperform 175B GPT-3 and 280B Gopher when trained on more data (1.4T tokens)\
  https://arxiv.org/abs/2203.15556
  https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications

- 2022 FLAN T5: Scaling Instruction-Finetuned Language Models\
  https://arxiv.org/abs/2210.11416
  https://github.com/google-research/FLAN/blob/main/flan/templates.py

- 2022 One Embedder, Any Task: Instruction-Finetuned Text Embeddings\
  It looks like it should be able to generate independently comparable embeddings, related against provided **zero-shot** task/domain, but I couldn't run the example.\
  https://arxiv.org/abs/2212.09741

- 2022 Efficient Training of Language Models to Fill in the Middle\
  FIM can be learned for free in auto-regressive models\
  https://arxiv.org/abs/2207.14255
