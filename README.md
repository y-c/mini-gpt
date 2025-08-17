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

### Phase 2: Make It Train (2 hours)
- [ ] Data preparation (character-level tokenization)
- [ ] Training loop with loss tracking
- [ ] Text generation with sampling
- [ ] Debug common training issues

### Phase 3: Experiment & Learn (2 hours)
- [ ] Learning rate ablations
- [ ] Architecture comparisons (depth vs width)
- [ ] Training dynamics visualization
- [ ] Memory profiling

### Future Extension: Multimodal (Optional)
- [ ] Simple vision encoder
- [ ] Interleaved image-text data
- [ ] Baby multimodal tasks

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
│   ├── 03_training_loop.ipynb
│   └── 04_experiments.ipynb
├── src/
│   ├── model.py          # GPT implementation
│   ├── train.py          # Training utilities
│   └── generate.py       # Text generation
├── data/
│   └── tiny_shakespeare.txt
├── checkpoints/          # Saved models
└── results/             # Logs, plots, findings
```

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

## 📝 Implementation Checklist

### Attention Mechanism
- [ ] Implement scaled dot-product attention
- [ ] Test with toy examples
- [ ] Visualize attention weights
- [ ] Understand masking for autoregressive models

### Multi-Head Attention  
- [ ] Project to Q, K, V
- [ ] Split heads correctly
- [ ] Concatenate and project output
- [ ] Verify shapes at each step

### Transformer Block
- [ ] Add feedforward network
- [ ] Implement layer normalization
- [ ] Add residual connections
- [ ] Test gradient flow

### Training
- [ ] Create data loader
- [ ] Implement teacher forcing
- [ ] Add gradient clipping
- [ ] Log training metrics

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

## 📚 Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide (reference only)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - For implementation details
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper (optional)

## 📈 Learning Milestones

- [ ] "I understand why we scale attention scores"
- [ ] "I can explain what each attention head learns"  
- [ ] "I know why we need position embeddings"
- [ ] "I can debug a transformer from scratch"

## 🤝 Contributing

This is a personal learning project, but feel free to:
- Open issues for questions
- Share your training results
- Suggest experiments

## 📜 License

MIT - Use this for your own learning!

---

**Remember**: The goal isn't to build the best GPT, it's to understand how GPTs work by building one yourself. Every bug is a learning opportunity!