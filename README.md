# char-rnn

Character level language model using RNN.

Reference: https://gist.github.com/karpathy/d4dee566867f8291f086

Data sources:
1) Leo Tolstoy's War and Peace (3.3 MB): [ref](http://cs.stanford.edu/people/karpathy/char-rnn/)
2) Sherlock Homes full text (3.4 MB): [ref 1](https://sherlock-holm.es/ascii/) / [ref 2](https://sherlock-holm.es/stories/plain-text/cnus.txt)
3) Shakespeare's texts (4.6 MB): [ref](http://cs.stanford.edu/people/karpathy/char-rnn/)
4) Linux kernel (6.2 MB): [ref](http://cs.stanford.edu/people/karpathy/char-rnn/)
5) Wikipedia text (100 MB): [ref](http://prize.hutter1.net/)
6) Entire Linux source code (474 MB): [ref 1](http://cs.stanford.edu/people/karpathy/char-rnn/) / [ref 2](https://github.com/torvalds/linux)
```
$ git clone https://github.com/torvalds/linux.git
$ cd linux
$ find . -name "*.[c|h]" | shuf | xargs cat > linux.txt
```
