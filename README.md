# Mini-GPT: Learn by Building

A hands-on project to learn transformer architecture by building a GPT model from scratch.

## 🎯 Project Goals

1. **Implement transformer architecture from first principles** - No copying, learn by coding each component
2. **Experience real training challenges** - Debug loss spikes, handle OOM errors, fix convergence issues  
3. **Build intuition through experimentation** - Understand why certain hyperparameters matter
4. **Create a foundation for multimodal models** - Extensible to vision-language models later

## 🚀 Learning Approach

**Philosophy**: Learn by building, not by watching. Every concept is learned through implementation and debugging.

## 📋 Project Phases

### Phase 1: Build Core Components (3-4 hours) ✅
- [x] Scaled dot-product attention ✅
- [x] Multi-head attention ✅
- [x] Transformer block with residual connections ✅
- [x] Full GPT architecture ✅

### Phase 2: Make It Train (2 hours) ✅
- [x] Data preparation (character-level tokenization) ✅
- [x] Training loop with loss tracking ✅
- [x] Text generation with sampling ✅
- [x] Debug common training issues ✅

**Training Results**:
- Started on laptop CPU (too slow)
- Migrated to Google Colab with Tesla T4 GPU
- Model: 384d, 6 heads, 6 layers (~10.7M parameters)
- Trained for 5000 steps (~49 minutes on GPU)
- Final loss: 1.11 train, 1.48 validation
- **Success**: Model generates Shakespeare-style dialogue!

### Phase 3: Experiments & Analysis (2-3 hours)
- [ ] Learning rate ablation (1e-4 to 1e-2)
- [ ] Batch size vs gradient accumulation study
- [ ] Architecture variations (depth vs width)
- [ ] Memory profiling per component

### Future Extension: Multimodal (Optional)
- [ ] Simple vision encoder
- [ ] Interleaved image-text data
- [ ] Baby multimodal tasks

## 🎓 Key Learnings

### Attention Mechanism Insights

1. **Core Intuition**: Attention is a learnable way to decide which parts of input to focus on
   - Queries ask "what am I looking for?"
   - Keys provide "what information is available?"
   - Values contain "what to actually use"

2. **Critical Implementation Details**:
   - Scaling by √d_k prevents gradient vanishing (discovered via experiment)
   - Causal masking enforces autoregressive property for generation
   - Multi-head design allows learning different types of relationships

3. **Findings from Implementation**:
   - Without scaling: attention scores std=8.03 → training fails
   - With scaling: std=1.00 → stable training
   - Attention weights visualization shows model literally learning what to "look at"

### Transformer Block Insights

1. **Residual Connections are CRITICAL**:
   - Without residuals: gradient norm = 0.000086 (vanished!)
   - With residuals: gradient norm = 211.58
   - **2.5 million times stronger gradient flow!**
   - This enables training very deep networks (GPT-3 has 96 layers)

2. **Component Interactions**:
   - Layer Norm stabilizes activations across features
   - Feed-forward adds "thinking" capacity between attention steps
   - Dropout prevents overfitting
   - Pre-LN architecture (normalize first) is more stable

## 💡 Lessons Learned: Local vs Cloud Training

### Local CPU Training (MacBook Air)
- **Speed**: ~0.5-1 iterations/second
- **Estimated time**: 5000 steps would take ~2-3 hours
- **Issues**: Too slow for practical experimentation
- **Lesson**: CPU training is only viable for tiny models or debugging

### Google Colab GPU Training (Tesla T4)
- **Speed**: ~2.2 iterations/second (with evaluation overhead)
- **Actual time**: 49 minutes for 5000 steps
- **Speedup**: ~4-5x faster than CPU
- **Free tier limitations**:
  - Disconnects when idle (~90 min)
  - Stops when laptop closes
  - Maximum 12-hour sessions
  - Need to keep tab active

### Key Takeaways
- Always use GPU for transformer training
- Colab is excellent for learning (free GPU!)
- Save checkpoints frequently
- Don't close laptop during Colab training

## 🎉 Progress Log

- **Session 1**: Implemented core attention mechanisms
  - ✅ Scaled dot-product attention 
  - ✅ Causal masking for autoregressive models
  - ✅ Multi-head attention with 8 heads
  - ✅ Discovered importance of √d_k scaling through experiments

- **Session 2**: Built transformer block
  - ✅ Feed-forward network with GELU activation
  - ✅ Layer normalization for stability
  - ✅ Residual connections (discovered 2.5M× gradient improvement!)
  - ✅ Complete transformer block with all components integrated

- **Session 3**: Completed full GPT
  - ✅ Positional encodings with sinusoidal patterns
  - ✅ Complete GPT architecture (~10.7M parameters)
  - ✅ Character-level tokenization
  - ✅ Training pipeline with gradient clipping

- **Session 4**: Successful training
  - ✅ Migrated from CPU to GPU (Google Colab)
  - ✅ Trained for 5000 steps in 49 minutes
  - ✅ Final loss: 1.11 train, 1.48 validation
  - ✅ Model generates coherent Shakespeare-style text!

## 📊 Results

**Sample Generation:**
```
Prompt: ROMEO:
Generated: ROMEO: O, be a pardon some life!
LUCIO: I had so much of him.
YORK: What, belike him? Hear, lords!
```

The model learned:
- Character dialogue format
- Shakespearean vocabulary
- Dramatic structure
- Character names and interactions

## 🎓 Success Criteria

- ✅ Working GPT that generates coherent text
- ✅ Documented training curves from at least 3 experiments
- ✅ One novel finding about training dynamics
- ✅ Understanding of memory/compute bottlenecks

## 🛠️ Tech Stack

- **PyTorch** - Core implementation
- **Matplotlib** - Visualizations
- **NumPy** - Data handling
- **Weights & Biases** (optional) - Experiment tracking

## 📁 Repository Structure

```
mini-gpt/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_attention_mechanism.ipynb
│   ├── 02_transformer_block.ipynb
│   ├── 03_gpt_model.ipynb
│   └── 04_training_gpt_colab.ipynb
├── data/
│   └── tiny_shakespeare.txt
├── checkpoints/          
│   └── shakespeare_gpt_complete.pt  # Trained model (41MB, 10.7M params)
└── results/             # Logs, plots, findings
```

## 🚀 Quick Start - Using the Trained Model

### Load and Generate Text (5 minutes)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# First, copy all the model classes from notebooks/04_training_gpt_colab.ipynb
# (GPT, TransformerBlock, MultiHeadAttention, etc.)

# Load checkpoint
checkpoint = torch.load('checkpoints/shakespeare_gpt_complete.pt')

# Restore tokenizer
chars = checkpoint['chars']
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = checkpoint['vocab_size']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Recreate model (must match training architecture!)
model = GPT(
    vocab_size=vocab_size,
    d_model=384,
    n_heads=6,
    n_layers=6,
    max_len=256,
    dropout=0.2
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text!
prompt = "ROMEO:"
context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0)

# Need to copy generate function from notebook too
# Then: generated = generate(model, context, max_new_tokens=100)
```

### Key Files
- `checkpoints/shakespeare_gpt_complete.pt` - Your trained model (41MB)
- `notebooks/04_training_gpt_colab.ipynb` - Has all the code
- Copy model classes and generate function from the notebook

## 🏃 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/mini-gpt.git
cd mini-gpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib

# Start with the first notebook
jupyter notebook notebooks/01_attention_mechanism.ipynb
```

## 📝 TODO: Next Steps

### Immediate Next Session
1. **Load trained model**:
   ```python
   checkpoint = torch.load('checkpoints/shakespeare_gpt_complete.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

2. **Phase 3 Experiments** (pick one):
   - Temperature ablation study
   - Architecture comparisons (depth vs width)
   - Different datasets (try modern English?)
   - Implement beam search

3. **Analysis Tasks**:
   - Visualize attention patterns
   - Study what each layer learned
   - Profile memory usage

### Toward Multimodal (MINT-1T path)
1. **Vision Module**:
   - Start with MNIST classifier
   - Add patch embeddings
   - Merge with text embeddings

2. **Simple Multimodal Tasks**:
   - "This digit is: [5]"
   - Image captioning on simple datasets
   - Text + image retrieval

3. **Infrastructure**:
   - Proper data pipeline
   - Multi-GPU training
   - Weights & Biases logging

### Project Ideas
- Fine-tune on specific authors
- Build a chatbot interface
- Train on code instead of Shakespeare
- Implement RLHF basics

## 🐛 Common Issues & Solutions

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Loss is NaN | Attention scores too large | Add scaling factor: 1/√d_k |
| Loss plateaus | Learning rate too low | Try increasing by 10x |
| Repeating tokens | Sampling temperature = 0 | Add temperature > 0 |
| OOM error | Batch size too large | Reduce batch size or use gradient accumulation |

## 📊 Experiments to Try

1. **Learning Rate Sweep**: Compare 1e-3, 1e-4, 1e-5
2. **Model Size**: 100K vs 1M parameters  
3. **Depth vs Width**: 4 layers × 128d vs 2 layers × 256d
4. **Position Encoding**: Learned vs sinusoidal

## 🎉 Minimum Viable Success

Your first goal: Generate *any* text that contains real words after training for 10 minutes on your laptop. Even "the the the cat" counts as success!

## 📚 Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide (reference only)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - For implementation details
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper (optional)

## 📈 Learning Milestones

- [x] "I understand why we scale attention scores"
- [x] "I can explain what each attention head learns"  
- [x] "I know why we need position embeddings"
- [x] "I can debug a transformer from scratch"

## 🤝 Contributing

This is a personal learning project, but feel free to:
- Open issues for questions
- Share your training results
- Suggest experiments

## 📜 License

MIT - Use this for your own learning!

---

**Remember**: The goal isn't to build the best GPT, it's to understand how GPTs work by building one yourself. Every bug is a learning opportunity!