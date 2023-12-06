# congestion
Congestion prediction models for CMA CGM

TODO IN GENERAL 

- try to develop an attention suited mechanism for congestion prediction (and number of boats) based on GAT
- search for needed try RNN models with a spatial graph part and to implement and test them
- think of appropriate adjacency in the network 

TODO CURRENT
 
- tokenization of congestion 
- transformer decoder part
- transformer decoder? number of boats part 
- part of adjacency matrix as a number of routes between terminals 
- simple decoder with wrappers 
- congestion batches part 
- debugging visualisation tools and (congestion visualisation)
- GPU config 
- Efficient fast beautiful dataclasses
- Generation from transformer 
- general pipeline for experience
- general debugging for models
- fine-tuning interface


REMARKS (should be on Git issues): 

- Efficient Batching in Embeddings 
- Self-Dropout of MLP in Decoder? 
- Positional Encodings to Values? 
- ohe-hot vectors labeling to try?