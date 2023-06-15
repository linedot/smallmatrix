## SMALL matrix multiplication performance ##

Multiplying 4x4 FP64 matrices. Currently only AVX/FMA

Building with icpx + papi:
```
/opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx -flto -Ofast -mavx2 -mfma -o smallmatrix smallmatrix.cpp performance_counters_papi.cpp -lpapi
```

Sample output:

```
~/SourceCode/test/smallmatrix (master ✘)✭ ᐅ ./smallmatrix 32 100000     
[...]
naive C algorithm finished in 454 cycles
FLOPS/CYCLE:     9.02
LD queue stalls: 0.00
L1D misses:      0.00
AVX/FMA algorithm finished in 595 cycles
FLOPS/CYCLE:     6.88
LD queue stalls: 249.00
L1D misses:      0.00
~/SourceCode/test/smallmatrix (master ✘)✭ ᐅ 
```

Results on AMD Ryzen 3950x:

| #Matrices | Iterations | autovec cycles | autovec LDQ stalls | manual cycles | manual LDQ stalls |
|-----------|------------|----------------|--------------------|---------------|-------------------|
| 32        | 100000     | 454            | 0                  | 595           | 249               |
| 64        | 100000     | 918            | 0                  | 1202          | 420               |
| 128       | 100000     | 1973           | 70                 | 2644          | 1242              |
| 256       | 100000     | 3929           | 86                 | 5271          | 2492              |
| 2048      | 10000      | 33248          | 1956               | 42855         | 19707             |
| 40000     | 1000       | 990772         | 75203              | 1043552       | 593170            |
| 800000    | 100        | 43325389       | 12194272           | 43899368      | 34863735          |

autovec is better for now, the "wrong" code is outperforming it though, so I need to figure out how to properly load the data.
