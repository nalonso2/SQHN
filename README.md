Code used for the paper 'A Sparse Quantized Hopfield Network for Online-Continual Memory'.

Python 3.7.6 and Pytorch 1.10.0 were used to generate results.

All datasets besides the TinyImagenet dataset are downloaded automatically via Pytorch. To download TinyImagenet see https://github.com/tjmoon0104/pytorch-tiny-imagenet?tab=readme-ov-file


To reproduce data run:

'main.py --test argument'


To reproduce plots run:

main.py --plot argument


The arguments for the various tests are below. 
These arguments are used to reproduce tests and plots.

associative memory comparisons: 'assoc_comp'

online-continual tests one hidden layer: 'OnCont-L1'

online-continual tests three hidden layer: 'OnCont-L3'

noisy encoding tests one hidden layer: 'nsEncode-L1'

noisy encoding tests one hidden layer: 'nsEncode-L3'

noisy encoding tests one hidden layer: 'recog'

architecture comparisons: 'arch compare'


For example, to reproduce the plots for the online-continual task with the one hidden layer models, run 

main.py --test OnCont-L1

followed by 

main.py --plot OnCont-L1


