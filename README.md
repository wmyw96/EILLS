# Environment Invariant Linear Least Squares

The script to reproduce the illustration section in our paper ``[Environment Invariant Linear Least Squares](https://projecteuclid.org/journals/annals-of-statistics/volume-52/issue-5/Environment-invariant-linear-least-squares/10.1214/24-AOS2435.full)''. We offer the implementations of brute force search and gumbel approximation at [brute_force.py](methods/brute_force.py) and [eills_gumbel.py](methods/eills_gumbel.py), respectively.

To cite our paper

```bibtex
@article{fan2024environment,
    AUTHOR = {Jianqing Fan and Cong Fang and Yihong Gu and Tong Zhang},
     TITLE = {Environment invariant linear least squares},
   JOURNAL = {The Annals of Statistics},
      YEAR = {2024},
    VOLUME = {52},
    NUMBER = {5},
     PAGES = {2268-2292},
      ISSN = {0090-5364},
       DOI = {10.1214/24-AOS2435},
      SICI = {0090-5364(2024)52:5<2268:EILLS>2.0.CO;2-7},
}
```

The remainings are the instructions to reproduce the results in the paper.

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

