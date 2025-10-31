# Analysis-Dynamic-F-Function

Source code for the paper titled "Analysis of a Dynamic F-Function: Full-round Integral Distinguisher for Lightweight Cipher SLIM-DDL"

## Source Code

### There are 14 files and 01 folder in this source code.

* `DDL_implementation.py`
* `DDL_Analysis.py`
* `SLIM_DDL_equivalent_round_keys.py`
* `SLIM_DDL_mask_scan.py`
* `SLIM_DDL_mask_test.py`
* `parse_and_plot_mask_results.py`
* `avg_zero_sum_plot.py`
* `DDL_all_outputs.csv`
* `Results_DDL_collisions.txt`
* `mask_executions_output.txt`
* `Resuls.txt`
* `README.md`
* `plots_mask_results`
* `LICENSE`

### Implementation of DDL

* `DDL_implementation.py`  implements the functionality of DDL.

### Analysis of DDL
* `DDL_Analysis` analyses the DDL input and output patterns. 
* Results are stored in `DDL_all_outputs.csv`.

### Equivalent Round Keys
* `SLIM_DDL_equivalent_round_keys` compute equivalent round keys for each round. Validates the results with 3 sets of plaintext and round keys. Sample output is stored in Results.txt

### Scanning various types of masks
* `SLIM_DDL_mask_scan.py` scans various types(weight based, sliding etc. ) of masks to search high probability zero sum masks. 

### Testing the masks
* `SLIM_DDL_mask_test.py` test a mask with various parameters. A sample run is as follows:

`python slim_mask_test.py --mask 0x2001 --trials 50000 --batch-size 1000 --rounds 32`
`Output:`
`Result: {'mask': 4097, 'mask_size': 2, 'trials': 16384, 'total data': 65536, 'zero_sum_count_mask': 8, 'zero_sum_count_random': 1, 'prob_zero_sum_mask': 0.00048828125, 'prob_zero_sum_random': 6.103515625e-05}
Elapsed: 0.03 s`

### Plot the results 
* `parse_and_plot_mask_results.py` plots the masked vs random zero sum counts.
* `avg_zero_sum_plot.py` plots average zero sum counts for masks. 

### Results
* All collisions in output are stored in `Results_DDL_collisions.txt`.
* All results of `SLIM_DDL_mask_test.py` are stored in `Results.txt`.
* All plots are stored in the folder `plots_mask_results`.
