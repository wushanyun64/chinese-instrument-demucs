# Concepts

## Why two sources if we only want one?

Demucs's data pipeline and augmentation assume the stems sum to the mixture. A literal
1-source model would violate this invariant. Instead we define `['chinese-flute', 'other']`
where `other = mixture − chinese-flute`. We keep the flute output and discard `other`.

## Synthetic data

Training mixtures are built on-the-fly by overlaying isolated flute recordings onto
flute-free background music:

```
mixture = gain_f × flute + gain_o × other
```

This gives effectively unlimited, diverse training data without needing full multitrack
recordings with isolated flute stems.

## Warm-start

We load pretrained `htdemucs` weights, drop the 4-source output head keys (shape mismatch),
and load the rest with `strict=False`. The encoder, decoder, and transformer layers
transfer their learned audio representations; only the source-dependent output projections
are reinitialized for our 2-source task.

## Sum-to-mixture invariant

For every track: `mixture.wav == chinese-flute.wav + other.wav` sample-exact.
We write `other` as the literal background used, and `mixture` as their exact sum.
Any drift corrupts training.
