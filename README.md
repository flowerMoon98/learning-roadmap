# learning-roadmap
brainlet trying to brain

## Roadmap Overview & Topics
### Foundations Refresher
Linear Algebra and Probability that actually matter for DL
Python/PyTorch for the dirty work
Project: Build Micrograd. Afterwards you'll build an MLP and train it
### Transformers
Tokenization, embeddings, self-attention, all the block diagram stuff
Pre-training paradigms: BERT/MLM vs GPT/CLM, and the why, how, and when
Project: Build a working mini-GPT from scratch

### Scaling and Training
How "scaling laws" actually predict performance (math)
Distributed training: Data, Tensor, Pipeline parallelism
Project: Spin up multi-GPU training with HuggingFace Accelerate. Make it run, see why things break, fix it

### Alignment + Fine-Tuning
RLHF/Constitutional AI
LoRA/QLoRA: parameter-efficient fine-tuning
Project: Implement LoRA from scratch. Plug it into a HuggingFace model and actually fine-tune on a use case


### Inference Optimizations
Inference optimization: FlashAttention, quantization, getting sub-second responses







You start here ..

- Karpathy's Micrograd series. <https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ> 
- Linear Algebra/Probability: 3Blue1Brown's videos.
- https://course.fast.ai/Lessons/lesson1.html
- https://www.youtube.com/@SebastianRaschka/playlists


You also read this ..
https://www.oreilly.com/library/view/ai-engineering/9781098166298/
https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/
https://www.oreilly.com/library/view/build-a-large/9781633437166/ (optional if you watch this ... <https://www.youtube.com/@SebastianRaschka/playlists> )



### Phases 1: Transformers

Intuition: 3Blue1Brown on Transformers/Attention. Jay Alammar's Illustrated Transformer. Watch, take notes, and re-watch if you need to.
Formal: Stanford CS224N Natural Language Processing with Deep Learning (the lectures, not just the slides).
Paper: "Attention Is All You Need". Don't read it yet if you haven't built the mental model above. Otherwise, you'll drown. READ ONLY ONCE COMFORTABLE WITH ALL THE ABOVE.
Hands-on: Karpathy's "Let's Build GPT" (eureka moment, you'll realize how simple all of it is).
Project: Reimplement a decoder-only GPT from scratch. Bonus points: swap in your own tokenizer, try BPE/SentencePiece.


### Phases 2: Scaling Laws & Training for Scale

LLMs got good through figuring out what to scale, how to scale it, proving it could scale, and showing that it actually works.
Papers: "Scaling Laws for Neural Language Models" (Kaplan et al), then "Chinchilla" (Hoffmann et al). Learn the difference.
Distributed Training: Learn what Data, Tensor, and Pipeline Parallelism actually do. Then set up multi-GPU training with HuggingFace Accelerate. Yes, you'll hate CUDA at some point. Such is life.
Project: Pick a model, run a small distributed job. Play with batch sizes, gradient accumulation. Notice how easy it is to run out of VRAM?


### Phases 3: Alignment & PEFT

Fine-tuning is not just a cheap trick. RLHF and PEFT are the reason you can actually use LLMs for real-world use cases.
RLHF: OpenAI's "Aligning language models to follow instructions" blog post, then Ouyang et al's paper. Grasp the SFT ➡️ Reward Model ➡️ RL pipeline. Don't get lost in PPO math too much.
CAI/RLAIF: Read Anthropic's "Constitutional AI".
LoRA/QLoRA: Read both papers, then actually implement LoRA in PyTorch. If you can't replace a Linear layer with a LoRA-adapted version, try again.
Project: Fine-tune an open model (e.g. gpt2, distilbert) with your own LoRA adapters. Do it for a real dataset, not toy text.


### Phases 4: Production

Inference Optimization: Read the FlashAttention paper. Understand why it works, then try it with a quantized model.


## Where To Learn Them
Below is what to read/watch for the this learning plan.
Math/CS Pre-Reqs
3Blue1Brown: Essence of Linear Algebra (YouTube)
MIT 18.06: Linear Algebra (Strang, OCW)
Deep Learning Book (Goodfellow)
PyTorch Fundamentals
Karpathy: Neural Networks Zero to Hero
PyTorch Learn the Basics
Zero to Mastery PyTorch
Transformers & LLMs
Attention Is All You Need (Vaswani et al)
3Blue1Brown: What is a GPT? (YouTube)
Jay Alammar: The Illustrated Transformer
Karpathy: Let's Build GPT
Stanford CS224N (YouTube Lectures)
Scaling & Distributed Training
Kaplan et al: Scaling Laws
Chinchilla Paper (Hoffmann et al)
HuggingFace Accelerate
Alignment & PEFT
OpenAI: Aligning LMs to Follow Instructions
Anthropic: Constitutional AI
LoRA: Low-Rank Adaptation
QLoRA
LightningAI: LoRA from Scratch
Inference
FlashAttention Paper



