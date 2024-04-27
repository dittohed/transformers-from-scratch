# transformers-from-scratch

My **implementation of a classical transformer in PyTorch** (for self-learning purposes). 
Based on http://nlp.seas.harvard.edu/2018/04/03/attention.html.

The classes in `transformer.py` and `layers.py` are defined according to the 
[scheme](./architecture.png).

### Examples

Run [main](./main.py) to see how the code works on a simple task of restoring
input at output.

### Further plans

- [ ] Add EOS token
- [ ] Test on some real task (like translation)

### Requirements

See [requirements.txt](./requirements.txt) (Python 3.10.12, CPU).

