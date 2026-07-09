# Hardware Notes

## Ryzen 3 + 8GB RAM

Recommended first run:

- `configs/micro.json`
- `block_size`: 256
- `batch_size`: 8
- CPU only

If memory is tight:

- close browser
- reduce `batch_size`
- reduce `block_size`
- use `configs/micro.json` before `configs/tiny.json`

## Model scale

| config | rough class | purpose |
|---|---:|---|
| micro | 10M-20M | first success |
| tiny | 30M-50M | better Japanese text continuation |

The first goal is text generation, not strong chat ability.
