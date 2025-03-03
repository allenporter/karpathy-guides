# gpt-2

Will reference GPT2 and GPT3 papaers.

# Notes

- Mapreduce for attention and MLP
- The field names in the state dict are all inferred from the model variable names
- "This is correct but inefficient implementation of sampling." What is efficient?
- Ran into problems using pytorch, likely involving transferring tensors across the CPU
- Had to turn off Code-assist
- Good refresher on loss expectation: Should be roughly -ln(1/token size)

## Performance log

- B=16, T=1024 - Default precision:
   380ms per step, 42k tok/sec
- bfloat16 - Did not increase performance for me.
- torch.compile()
  - 139ms, 118k tok/sec. 2.3x speedup
