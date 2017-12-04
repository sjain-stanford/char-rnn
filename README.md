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
7) Arxiv abstracts (46 kB): [ref](https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/data/arvix_abstracts.txt)
```
$ git clone https://github.com/torvalds/linux.git
$ cd linux
$ find . -name "*.[c|h]" | shuf | xargs cat > linux.txt
```


min-char-rnn output

```
iter 2500, loss: 64.751443
-----
 cam's we ofowkopris, coulst ,
A auss hong t'an af youkz, bes,
Rty, ic-
H Cout, :of woucd ungr pgivow: cave your sircer hics, I secese, Iuc uitf son court soo natt hure sor,
Br hory med Wiun:

CIRUTUS:
 -----
iter 2600, loss: 64.002116
-----

fhpooj shs, kyifs thom it uotmurd'd se care.

BNUS:
Toumhe t andius; t, yoor, jon? bp yethes bly youias aur, i, io, d
wns;t'gorn thavanr. wocigivepnoo rresptin hiuvteemor he'envone hinsler rowvorame
 -----
iter 2700, loss: 63.210558
-----
 el hank the weheld wowthad gltto that mo hathich ve phe pomy, wupe wore the sle the thintdatis nsm horl:
Have yhe yory ul p er,
th't to thit; yor or. ftS
Brimiot pames pes hos entaohet petire me hher
 -----
```

```
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
