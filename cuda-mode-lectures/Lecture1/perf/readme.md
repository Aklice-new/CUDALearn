# Perf

## ncu

通过ncu对.py进行profile。
```shell
ncu -o output_kernel python file_name.py

```
然后把output_kernel保存的结果可以直接拖进Nsight Compute中进行分析。