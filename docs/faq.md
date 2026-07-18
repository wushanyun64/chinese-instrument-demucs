# FAQ

## Can it separate other instruments?

Not yet. This is a single-target model trained specifically for Chinese flute.
To separate other instruments, you'd need to retrain with different data.

## How much data do I need?

Plan's guidance: flute clip diversity matters more than raw hours. Aim for multiple
instruments, players, keys, and recording conditions. Background pool should be
roughly an order of magnitude larger and stylistically varied.

## Why two sources if I only want one flute stem?

See [Concepts](concepts.md). Demucs's pipeline assumes stems sum to the mixture.
A 1-source model would break that invariant. We model `['chinese-flute', 'other']`
and discard `other`.

## Can I use this without a GPU?

Training requires a GPU. Inference works on CPU but is much slower.
