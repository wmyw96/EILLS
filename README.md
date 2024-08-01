# Environment Invariant Linear Least Squares

The script to reproduce the illustration section in our paper ``Environment Invariant Linear Least Squares''.


## Reproduce the results in Section 5

First conduct simulations and save the results in ``~/eills_demo.npy'' using the following command:

```bash
python eills_demo.py
```

Then the results in Fig. 3 can be presented running the following commands

```bash
python regularization_path.py
python eills_demo_vis11.py
python eills_demo_vis12.py
```

For the comparsion with other invariance learning methods, we also run simulations and save the results using the command

```bash
python eills_demo2.py
```

We run the following commands to generate the results in Fig. 4.

```bash
python eills_demo_vis21.py
python eills_demo_vis22.py
python eills_demo_vis23.py
```

## Reproduce the results in the supplemental material

For Fig. 3, run single gumbel approximated EILLS using

```bash
python test_eills_gb.py --mode 3 --seed 1
```

For Fig. 4, first run the simulations and then visualize the saved results

```bash
python unit_test_eills.py --mode 0
python unit_test_eills.py --mode 1
```

