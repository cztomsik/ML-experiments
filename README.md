My personal learning space. Nothing to see here, except maybe for the links below.

## Interesting Papers (mostly NLP)

- **2017 Unsupervised Sentiment Neuron**\
  First? proof that LSTM network can learn something useful when trained to "just" predict next token.\
  This confirms that **prediction implies understanding**, which OFC makes sense but we were not sure at the time.\
  Implications of this are huge, because from this point we only need to find more efficient way of training or mode data/compute to get better results.\
  https://openai.com/blog/unsupervised-sentiment-neuron/

- **2017 Attention is all you need**\
  An architecture (transformer) which can use long-range dependencies (like RNN) but can be trained in a very efficient way.\
  Instant SOTA in machine-translation, but basically from this point, every huge leap is based on this paper.\
  https://arxiv.org/abs/1706.03762

- **2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**\
  Encoder-only transformer, SOTA in NLP inference.\
  https://arxiv.org/abs/1810.04805

- **2019 RoBERTa: A Robustly Optimized BERT Pretraining Approach**\
  Same BERT model, with better training: masking-only, dynamic masking, bigger batches, more data\
  New SOTA\
  https://arxiv.org/abs/1907.11692

- **2019 Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**\
  Sentence to vector. Vectors can be compared independently.\
  https://arxiv.org/abs/1908.10084

- **2019 T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**\
  If you train transformer with "prefix" task, it will get better at learning new tasks.\
  T5 is very easy/eager to fine-tune for new tasks with very little data, you just need to follow the same "pattern" of `task>example>result` which it recognizes.
  https://arxiv.org/abs/1910.10683

- **2020 GPT-3: Language Models are Few-Shot Learners**\
  Large enough model is capable of predicting the likely output from few examples in its prompt without any fine-tuning\
  They literally say they took the same GPT-2 model, scaled it up, and new capabilities just "emerged"\
  https://arxiv.org/abs/2005.14165

- **2020 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**\
  Transformers can be used for image recognition, you just need to slice image to 16x16 "tokens" and feed it through "embedding" matrix/linear-layer and that's it, instant leap in model performance.\
  https://arxiv.org/abs/2010.11929

- **2020 Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**\
  Transformer rewritten as RNN can be used for speeding-up the inference.\
  This is very cool but it didn't catch up, maybe because these models were always performing slightly worse?\
  https://arxiv.org/abs/2006.16236

- **2021 FLAN: Finetuned Language Models Are Zero-Shot Learners**\
  This is sort of similar to the Instruct-GPT paper, but without RLHF.\
  https://arxiv.org/abs/2109.01652
  https://github.com/google-research/FLAN/blob/main/flan/templates.py

- **2021 An Attention Free Transformer**\
  Alternative Transformer design\
  https://arxiv.org/abs/2105.14103
  - **AFT-inspired time-mixing variation (WKV) with RNN/GPT mode, very fast\**\
    https://github.com/BlinkDL/RWKV-LM

- **2021 (ALiBi) Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation**\
  Alternative position embedding\
  https://arxiv.org/abs/2108.12409

- **2022 FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**\
  Very cool, but it needs custom CUDA kernel.\
  https://github.com/HazyResearch/flash-attention
  https://arxiv.org/abs/2205.14135

- **2022 Chinchilla**\
  50B model can outperform 175B GPT-3 and 280B Gopher when trained on more data (1.4T tokens)\
  https://arxiv.org/abs/2203.15556
  https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications

- **2022 FLAN T5: Scaling Instruction-Finetuned Language Models**\
  Continuation of the previous FLAN paper. More tasks, more models, etc.\
  Very cool, T5 got better at pretty much everything.\
  https://arxiv.org/abs/2210.11416
  https://github.com/google-research/FLAN/blob/main/flan/templates.py

- **2022 One Embedder, Any Task: Instruction-Finetuned Text Embeddings**\
  It looks like it should be able to generate independently comparable embeddings, related against provided **zero-shot** task/domain, but I couldn't run the example.\
  https://arxiv.org/abs/2212.09741

- **2022 Efficient Training of Language Models to Fill in the Middle**\
  FIM can be learned for free in auto-regressive models\
  https://arxiv.org/abs/2207.14255

- **2022 pNLP-Mixer: an Efficient all-MLP Architecture for Language**\
  Embedding-free MLP-mixer, which takes MinHash of per-word tokens through bottleneck mapping layer. Then it's just MLP-mixer. Small, fast, okish results, scales down nicely. Fixed-size.
  https://arxiv.org/abs/2202.04350
  https://github.com/mindslab-ai/pnlp-mixer

- **2022 VCT: A Video Compression Transformer**\
  Lossless compression by entropy coding result from lossy transformer. Very cool.\
  https://arxiv.org/abs/2206.07307

- **2023 Toolformer: Language Models Can Teach Themselves to Use Tools**\
  LLMs can use in-context learning to use external tools/APIs\
  https://arxiv.org/abs/2302.04761
