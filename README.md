# The Transformer, Number by Number

Eleven parts that build the transformer one hand-computed number at a time. Every matrix is
small enough to check with a pencil, and every number on every page comes from the same
three words: `I love LLMs`, translated into French.

**[Read the series →](https://haoma7.github.io/TTNN-Transformer-Tutorial/)**

## Why another transformer tutorial

You have seen the diagram and the attention formula, and neither one told you what actually
happens to a sentence. This series answers that by running the arithmetic in front of you.
Where other explanations say "attention weights the values," this one prints the weight
matrix, prints the value matrix, multiplies them, and shows you the result.

```
X   = [[0,0,1,1],     W^Q = [[ 0,-1,1, 1],    W^K = [[ 0,-1,0, 1],    W^V = [[0,0,1,0],
       [0,1,0,1],            [ 0, 1,0, 0],           [-1, 1,0, 0],           [1,0,0,1],
       [1,0,0,1]]            [ 1, 0,0,-1],           [ 1,-1,1,-1],           [0,1,0,0],
                             [ 1, 1,0, 1]]           [ 1, 1,0, 1]]           [1,0,0,0]]
```

That is the whole model. Part 1 turns it into attention weights. Part 5 puts it through an
encoder block. Parts 6 to 8 build the decoder and translate the sentence. Part 9 trains it
and Part 10 generates from it.

It is also written for readers whose first language is not English: short sentences, one
idea at a time, and a fixed vocabulary that never changes words for the sake of elegance.

You need matrix multiplication and basic calculus. You do not need prior deep learning
experience. Each part is one self-contained HTML file — open it in a browser, no build step,
no server, works offline.

## The series

Read in order. Each part starts from where the last one stopped.

**[Part 0 — How a Sentence Becomes Numbers](./TTNN-Part0-Tokenisation.html)**
Eleven characters become tokens, tokens become id numbers, and id numbers become rows.
Byte-pair merges counted by hand, the three markers, and the embedding table.

**[Part 1 — Self-Attention](./TTNN-Part1-Self-Attention.html)**
What recurrence costs, what Query, Key and Value each are, and the whole attention formula
worked out on three words.

**[Part 2 — Multi-Head Attention](./TTNN-Part2-Multihead-Attention.html)**
One head gives every token a single budget to spend. Why that is the wrong shape for
language, what splitting splits, and how many heads to use.

**[Part 3 — Why a Transformer Cannot See Word Order](./TTNN-Part3-Positional-Encoding-1.html)**
Shuffle the sentence and the output rows only move with it. The proof that order is lost,
why no later layer can put it back, and why a plain counter fails.

**[Part 4 — How Positional Encoding Works](./TTNN-Part4-Positional-Encoding-2.html)**
The sine and cosine formula worked by hand, where it is added, why it lets a model reason
about distance, and the rotary version that current models use instead.

**[Part 5 — What Attention Needs Around It](./TTNN-Part5-Encoder-Block.html)**
Attention alone cannot answer a yes-or-no question about a row. The three things wrapped
around it — the residual addition, layer normalisation, the feed-forward network — and the
finished encoder block.

**[Part 6 — How the Decoder Writes](./TTNN-Part6-Decoder-Inference.html)**
A row leaving the stack is not a word. The output head that turns one row into a word, the
loop that feeds the word back in, and the two markers the loop needs.

**[Part 7 — Where the Encoder Stack Meets the Decoder Stack](./TTNN-Part7-Cross-Attention.html)**
Cross-attention is the step that carries the encoder's rows into the decoder. One such step
worked by hand, the three sublayers of a decoder block, and the whole transformer in one figure.

**[Part 8 — What a Row Is Not Allowed to See](./TTNN-Part8-Masks.html)**
On the second run the multiplication produces one score the rules forbid. The causal mask
that stops it counting, and the padding mask for sentences of different lengths sharing one batch.

**[Part 9 — How the Weights Are Corrected](./TTNN-Part9-Training.html)**
Every weight so far was written down by hand. Training is the correction: a sentence
somebody already wrote becomes a number, the number becomes a direction, and the direction
changes every weight.

**[Part 10 — Choosing the Word](./TTNN-Part10-Generation.html)**
The model gives a probability for every word in the vocabulary, and something has to pick
one. Greedy choosing, sampling, temperature, top-k and top-p, beam search, and what caching
saves.

## Contributing

The most useful contributions here are the smallest ones. If you read a paragraph twice and
still did not follow it, that is a bug in the writing — and telling us where you stopped is
the whole contribution. You do not need to propose a fix.

Open an issue for any of these:

- **You got lost.** A sentence you had to read twice, a word used before it was defined, a
  pronoun you could not resolve. If English is not your first language, your report is worth
  more than a native speaker's, because this series is written for you.
- **A number looks wrong.** Recompute a step and tell us what you got. Published values have
  been corrected this way before.
- **Something is broken.** A dead link, a figure numbered wrong, labels that overlap on your
  screen, a formula that renders as raw LaTeX.
- **A figure would explain it better.** Describe it, or draw it.

Rewriting a section or translating a part is welcome too — open an issue first, so we can
talk it through. Sections carry constraints that are not obvious from reading them, and it
would be a shame to waste your time.

### Three things to know before editing

**The vocabulary is fixed.** These were chosen deliberately. A second-language reader treats
a new word as a new thing, so please do not vary them for elegance.

| Term | Means | Not |
|---|---|---|
| **position** | where a word sits in the sentence, from 0 | slot, index, timestep |
| **slot** | where a number sits inside one vector, from 0 | dimension, position, cell |
| **embedding vector** | what a word means, before position is added | word vector, token vector |
| **block** | one encoder block or one decoder block | tile, rectangle, grid square |
| **encoder stack** / **decoder stack** | six blocks, one on top of another | *stack* on its own |
| **token** | one piece of text the model has a row for | *word*, where the two differ |

**Clarity beats brevity.** Prefer wordy to unclear. British spelling in prose, `-ise` not
`-ize`. Each part numbers its own sections and figures from 1.

**Open the page in a browser before you open the pull request.** There is no build step and
no CI, so nothing else will catch a formula the HTML parser ate or a label that collides
after a font falls back. Read the passages you changed, at the width a reader would use, in
both light and dark. If you changed a number, show your working in the pull request. One
concern per pull request, and use the smallest edit that works.

## The code track

Planned, not yet published. Every part will get a PyTorch exercise that implements what that
part teaches, so a reader who finishes the series has written the whole transformer one piece
at a time. Each exercise is a standalone file with its own weights, and each one asserts that
your code reproduces the numbers printed on the page — digit for digit. That test only exists
because the series spent eight parts printing every intermediate number.

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Teaching from it,
translating it, and quoting it in your own notes are all fine, as long as you credit the
source and share any derivative on the same terms. Selling it, or putting it inside a paid
course or product, is not. For commercial use, open an issue.

The architecture follows Vaswani et al., *Attention Is All You Need* (2017).
