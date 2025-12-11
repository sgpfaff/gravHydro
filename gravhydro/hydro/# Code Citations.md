# Code Citations

## License: BSD_3_Clause
https://github.com/jbailinua/gravhopper/tree/f9197c31714237a04a0613115b131819b9598007/gravhopper/gravhopper.py

```
Rax = np.arange(0.001*Rd_kpc, 10*Rd_kpc, 0.01*Rd_kpc)
    R_cumprob = Rd_kpc**2 - Rd_kpc*np.exp(-Rax/Rd_kpc)*(Rax+Rd_kpc)
    R_cumprob /= R_cumprob[-1]
    probtransform
```

