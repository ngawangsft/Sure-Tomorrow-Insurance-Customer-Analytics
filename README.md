# Insurance Customer Analysis — Sure Tomorrow

## Overview
Sure Tomorrow insurance wants us to help them out with a few data problems. They wanted to know if machine learning could make their marketing smarter, predict which customers might file claims, and most importantly, keep customer data safe without breaking their models. So we rolled up our sleeves and tackled four tasks using linear algebra, k‑nearest neighbors, and linear regression.

## Business Problem
Insurance companies have tons of customer data, but using it wisely is tricky. They need to:
- Find customers similar to their best ones for targeted marketing.
- Predict who's likely to file a claim (and how many claims).
- Protect personal info so that even if data leaks, privacy stays intact—but models still work.

**Our goal:** Build and test algorithms for similarity search, classification, regression, and data obfuscation, making sure privacy measures don't mess up prediction quality.

## Highlights & Key Results
We broke the project into four tasks, and here's what we found:

- **Task 1 – Finding Similar Customers**  
  Scaling matters a ton for k‑NN. Without scaling, income (which has big numbers) completely dominates distance calculations—neighbors ended up with the same income but random other features. After scaling with `MaxAbsScaler`, all features got a fair shot, and distances shrank from ~6–14 down to ~0.003–0.03. Euclidean and Manhattan gave almost identical neighbor lists, so either metric works fine.

- **Task 2 – Predicting If a Customer Will Get Benefits**  
  We built a k‑NN classifier and compared it to a dummy model that just guesses randomly. The dummy model's F1 score maxed out at 0.20. But with scaled data and k=3, our k‑NN hit **F1 = 0.95** – a massive improvement. Scaling was crucial here too; without it, performance tanked.

- **Task 3 – Predicting the Number of Benefits**  
  We coded our own linear regression from scratch (matrix math). It gave an RMSE of **0.34** and R² of **0.43**, not perfect, but decent for explaining about 43% of the variation. Scaling didn't change these numbers because linear regression just adjusts its weights. We even tested it on a new customer (gender=1, age=35, income=50k, family=2) and got a prediction of ~0.29 benefits.

- **Task 4 – Hiding Customer Data Without Hurting Models**  
  We obfuscated the numerical features by multiplying them by a random invertible matrix (call it `P`). Then we proved—both on paper and in code, that linear regression predictions stay exactly the same. The weights just transform to `w_P = P⁻¹ w`, and `X P w_P = X w`. So RMSE and R² are identical. That means Sure Tomorrow can safely share transformed data for analysis without losing any predictive power, keeping customers' info private.

### Quick Summary Table

| Task | What We Did | Key Result |
|------|-------------|------------|
| 1. Similar Customers | k‑NN with/without scaling | Scaling essential; neighbors make sense only after scaling |
| 2. Benefit Classification | k‑NN vs dummy model | F1 = 0.95 (k=3, scaled) vs 0.20 dummy |
| 3. Benefit Regression | Linear regression (our own) | RMSE = 0.34, R² = 0.43 |
| 4. Data Obfuscation | Multiply features by matrix `P` | RMSE & R² unchanged – privacy without loss |

## Final Thoughts
- Scaling is a must for distance‑based algorithms like k‑NN. Without it, big numbers dominate the little ones.
- k‑NN can be amazingly accurate for predicting insurance claims—way better than random guessing.
- Linear regression is solid but leaves a lot unexplained; there's room for improvement.
- Obfuscation with an invertible matrix is a slick way to protect privacy while keeping models happy. It's mathematically guaranteed to work for linear regression.

## Tools & Technologies
We used a bunch of Python goodies:
- `pandas`, `numpy` for data wrangling
- `seaborn`, `matplotlib` for quick plots
- `sklearn.neighbors` for k‑NN (both nearest neighbors and classifier)
- `sklearn.linear_model` for comparison, but we also built our own linear regression
- `sklearn.preprocessing` for scaling
- `sklearn.metrics` for RMSE, F1, etc.
- `sklearn.model_selection.train_test_split` to split data

## Dataset
The insurance data lives here:  
[insurance_us.csv](https://practicum-content.s3.us-west-1.amazonaws.com/datasets/insurance_us.csv)

Just drop it into a `data/` folder after downloading.

## How to Run
1. Clone or download the repo.
2. Make sure `insurance_us.csv` is inside the `data/` folder.
3. Open `insurance_analysis.ipynb` in Jupyter Notebook or VS Code.
4. Run all cells—everything's set up to reproduce the analysis and results for all four tasks.