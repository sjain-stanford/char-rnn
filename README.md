# char-rnn

Character level language model using RNN.

Reference: https://gist.github.com/karpathy/d4dee566867f8291f086 https://github.com/karpathy/char-rnn

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


min-char-rnn output
```
I all come me.

TRWARD:
I saining?

PERMIAN:
She, I wills my lib, a for thou, yet
Benich,
Detixting
For a martyan this thee:
 -----
iter 7206300, loss: 44.407445
-----
 Vork'ty tword,
And a so the inseinge leve
Compolt and Cay, cosit wither would iset to that nies there honour of is wourger, myet.
If think your me!

Sidge--
Wisefray I a mest came tay untise it,
A me
 -----
iter 7206400, loss: 44.328364
-----
  thinking and they smears are dints my lymer' or for the shile'd well that I Ho'en you,
The poyas's me it Rest wized to dnabyour, will but'a?

Neghnee larget.

DUKE ANDIK:
Dors.

GLOUCESTER:
What!
You
 -----
iter 7206500, loss: 44.499381
-----

Thosowarth'
Acolly you quanss.

TRAILE:
To for him loved oftoued?
Rungtices I slight arree;
I beace her you well, arce have desent, for in sees papest your teave
sholl thistefays, sir: priemace Arros
 -----

```
