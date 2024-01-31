Code used for the paper 'A Sparse Quantized Hopfield Network for Online-Continual Memory'.

Run with Python 3.7.6 and Pytorch 1.10.0.

All datasets besides the TinyImagenet dataset are downloaded automatically via Pytorch. To download TinyImagenet see https://github.com/tjmoon0104/pytorch-tiny-imagenet?tab=readme-ov-file


To reproduce data from a training run/experiment:

<code>main.py --test argument</code>


To reproduce plots for a training run/experiment:

<code>main.py --plot argument</code>


Here are the arguments for the various experiments
used to reproduce tests and plots:

associative memory comparisons: <code>assoc_comp</code>

online-continual tests one hidden layer: <code>OnCont-L1</code>

online-continual tests three hidden layer: <code>OnCont-L3</code>

noisy encoding tests one hidden layer: <code>nsEncode-L1</code>

noisy encoding tests one hidden layer: <code>nsEncode-L3</code>

noisy encoding tests one hidden layer: <code>recog</code>

architecture comparisons: <code>arch compare</code>


For example, to reproduce the plots for the online-continual task with the one hidden layer models, run 

<code>main.py --test OnCont-L1</code>

followed by 

<code>main.py --plot OnCont-L1</code>


