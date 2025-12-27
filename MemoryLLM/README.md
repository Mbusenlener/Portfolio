# Latent Memory-Augmented LLM
This model integrates latent memory tokens that are compressed from the hidden states of previously encoded forward-passes into decoding. Specifically, Each transformer layer maintains a pool of latent memories that is selectively observed by the current hidden states via cross-attention. During memory encoding, hidden states at each layer are compressed into K memory vectors and appended to the pool. Memory-attention during decoding scales linearly in complexity with respect to the number of memories in the pool, significantly reducing inference costs over long contexts. Each transformer layer maintains a pool of latent memories that is selectively observed by the current hidden states via cross-attention. During memory encoding, hidden states at each layer are compressed into K memory vectors and appended to the pool. Memory-attention during decoding scales linearly in complexity with respect to the number of memories in the pool, significantly reducing inference costs over long contexts.

## Code Map
* train_memformer.py: Primary training script defining the model architecture and training workflow.
* evaluate.py: Primary evaluation script that evaluates the memory augmented model on unseen HotpotQA samples and plots the results.
* report.pdf: Detailed descriptions of the methodology and findings of this project.

## Results
This model is evaluated on reading comprehension using the HotpotQA dataset. Experimental results show that the latent memory-augmented model
outperforms the In-Context baseline by 22% while compressing contexts by 8x. Both the baseline and memory-augmented model use Qwen2.5-1.5B as the base architecture.




