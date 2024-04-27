# transformers-from-scratch

My implementation of a classical transformer in PyTorch (for self-learning purposes). 
Based on http://nlp.seas.harvard.edu/2018/04/03/attention.html.

The classes in `transformer.py` and `layers.py` are defined according to the 
[scheme](./architecture.png).

### Examples

Run [main](./main.py) to see how the code works on a simple task of restoring
input short integer sequences at output:

```
Epoch [1/8]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:17<00:00,  1.11s/it, loss=1.91]
=== Inference demo ===
Source: [1, 4, 3, 7, 8, 2]
Prediction: [1, 3, 4, 4, 4, 2]

Epoch [2/8]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:17<00:00,  1.11s/it, loss=1.42]
=== Inference demo ===
Source: [1, 4, 3, 7, 8, 2]
Prediction: [1, 4, 8, 2, 3, 7]

...

Epoch [8/8]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:13<00:00,  1.19it/s, loss=0.405]
=== Inference demo ===
Source: [1, 4, 3, 7, 8, 2]
Prediction: [1, 4, 3, 7, 8, 2]
```

### Further plans

- [ ] Add EOS token
- [ ] Test on some real task (like translation)

### Requirements

See [requirements.txt](./requirements.txt) (Python 3.10.12, CPU).

