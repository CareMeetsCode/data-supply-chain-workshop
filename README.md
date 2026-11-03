# Building Fairness Into the Data Supply Chain

**AfroTech 2026 workshop · Waverly Rose Brim**

One lab value, creatinine, flows through an equation, into a model, and out to
a prescribing decision. This workshop follows it the whole way, using real CDC
NHANES data with linked mortality outcomes, and asks at every step what a
standard check would have missed.

**Workshop website:** <https://datasupplychain.waverlyrose.io>. Each step's page
explains the ideas; the notebooks below are where you run them.

## Run the notebooks

Each notebook opens in Google Colab and installs everything it needs in its
first cell. No setup on your machine.

| Step | Notebook | Open |
| --- | --- | --- |
| 2 | The Impact of One Lab Value | [Open in Colab](https://colab.research.google.com/github/CareMeetsCode/data-supply-chain-workshop/blob/afrotech-2026/notebooks/02_One_Lab_Value.ipynb) |
| 3 | Act I: Inherited Decisions | [Open in Colab](https://colab.research.google.com/github/CareMeetsCode/data-supply-chain-workshop/blob/afrotech-2026/notebooks/03_Inherited_Decisions.ipynb) |
| 4 | Machine Learning Missteps | [Open in Colab](https://colab.research.google.com/github/CareMeetsCode/data-supply-chain-workshop/blob/afrotech-2026/notebooks/04_Machine_Learning_Missteps.ipynb) |
| 5 | Act II: The Model Blind to Harm | [Open in Colab](https://colab.research.google.com/github/CareMeetsCode/data-supply-chain-workshop/blob/afrotech-2026/notebooks/05_Model_Is_Blind.ipynb) |
| 6 | Clinical Decisions at the Threshold | [Open in Colab](https://colab.research.google.com/github/CareMeetsCode/data-supply-chain-workshop/blob/afrotech-2026/notebooks/06_Clinical_Decisions.ipynb) |
| 7 | How Big Is This Really? | [Open in Colab](https://colab.research.google.com/github/CareMeetsCode/data-supply-chain-workshop/blob/afrotech-2026/notebooks/07_How_Big.ipynb) |
| 8 | The Second Failure | [Open in Colab](https://colab.research.google.com/github/CareMeetsCode/data-supply-chain-workshop/blob/afrotech-2026/notebooks/08_Second_Failure.ipynb) |
| 9 | The Hidden Harm Challenge | [Open in Colab](https://colab.research.google.com/github/CareMeetsCode/data-supply-chain-workshop/blob/afrotech-2026/notebooks/09_Hidden_Harm_Challenge.ipynb) |

Each notebook stands on its own: you can open any step without running the
ones before it. Step 9 runs faster on a GPU (**Runtime → Change runtime type →
T4 GPU**), but works on CPU with the same results.

### Run them locally instead

```bash
python -m pip install "data-supply-chain-workshop==0.1.2" jupyter
jupyter notebook notebooks/
```

Python 3.11 or newer. The package includes the analysis code and the NHANES
cohort the notebooks use.

## The code

[`src/data_supply_chain/`](src/data_supply_chain/) is the source of the
[`data-supply-chain-workshop`](https://pypi.org/project/data-supply-chain-workshop/)
package, version 0.1.2: the code the notebooks install and run. It is here so
you can read it. To run it, install the package as shown above.

If you are reading the code, a good order is:

| Module | What it does | Used in step |
| --- | --- | --- |
| `egfr.py` | The eGFR equations (2009 race-based, 2021 race-free), CKD stages, metformin tiers, and who crosses each threshold | 2, 3, 4, 6, 7 |
| `decisions.py` | The clinical decision layer: which patients' metformin tier changes, in which direction | 6, 7 |
| `models.py` | The mortality model, trained and evaluated on a fixed train/test split | 5, 8, 9 |
| `audit.py` | Subgroup audits, calibration error, and the ship/don't-ship gate | 5, 8, 9 |
| `survey.py` | Survey-weighted national estimates and their confidence intervals | 7 |
| `search.py` | The subgroup search for the Hidden Harm Challenge, with a permutation test that accounts for how many groups were searched | 9 |
| `hidden_harm_visuals.py` | The step 9 charts | 9 |
| `bootstrap.py` | `cold_start()` and `reset()`: the shared starting point every notebook loads | all |
| `data.py`, `preprocess.py` | Loading the cohort and preparing model features | all |
| `figures.py` | The eGFR and reclassification figures | 2, 4 |
| `findings.py`, `paths.py` | The record of published results behind the website (the notebooks' final "Record" cell) | all |

## Data

The cohort is built from the CDC National Health and Nutrition Examination
Survey (NHANES), cycles 1999–2000, 2001–2002, and 2003–2004, with the NCHS
Linked Mortality File. NHANES is public-domain US government data:
<https://www.cdc.gov/nchs/nhanes/>.

## License and citation

- **Code** (`src/`): [Apache License 2.0](LICENSE)
- **Workshop content** (notebook text, figures, website prose): [CC BY 4.0](LICENSE-CONTENT.md)

To cite the workshop, see [`CITATION.cff`](CITATION.cff), or use GitHub's
"Cite this repository" button.

---

Created by [Waverly Rose Brim](https://waverlyrose.io/) · Published by
[Care Meets Code](https://github.com/CareMeetsCode)
