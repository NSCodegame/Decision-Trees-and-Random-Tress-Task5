# ❤️ Predicting Heart Disease with Decision Trees & Random Forests

> Can we teach machines to predict heart disease? In this project, we build intelligent tree-based models that learn from real medical data and make accurate predictions — all explained in a way that’s beginner-friendly and practical.

---

## 🧠 Why This Project?

Heart disease is one of the leading causes of death globally. Early detection can save lives.  
In this project, we explore how machine learning — specifically **Decision Trees** and **Random Forests** — can help predict whether a person is at risk of heart disease using clinical data.

This project is part of **Task 5** in my machine learning journey, where I learned:

- How tree-based models work
- How to avoid overfitting
- How to interpret feature importance
- How to validate model performance

---

## 📁 Dataset Overview

- **Filename**: `heart.csv`
- **File Path**: `C:\Users\Nishtha Singh\Downloads\heart.csv`
- **Target Variable**: `target`
  - `1` → Patient has heart disease
  - `0` → No heart disease

The dataset contains medical attributes like:

- Age, Sex, Chest Pain Type (`cp`)
- Resting Blood Pressure (`trestbps`)
- Cholesterol (`chol`), Max Heart Rate (`thalach`)
- Fasting Blood Sugar (`fbs`), Exercise-induced Angina (`exang`)
- And more...

---

## 🧰 Tech Stack & Tools

| Tool         | Purpose                        |
|--------------|--------------------------------|
| `pandas`     | Load and process data          |
| `matplotlib` + `seaborn` | Visualize data & results |
| `scikit-learn` | Build ML models               |
| `plot_tree` | Visualize decision logic        |

---

## 🚦 What Did I Build?

### 1. 🧪 Data Loading & Cleaning
- Loaded the dataset using `pandas`
- Checked for missing values (none found)
- Split the data into features (`X`) and labels (`y`)

### 2. 🔍 Train-Test Split
Used an 80-20 split to separate training and testing data.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
````

---

### 3. 🌳 Decision Tree Classifier

* Trained a **Decision Tree**
* Visualized the tree structure
* Tuned `max_depth=4` to reduce overfitting
* Evaluated performance with accuracy, confusion matrix & classification report

**Visualization Output Example:**
![Decision Tree](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)
*Note: Replace with your actual tree image or use `plot_tree()`.*

---

### 4. 🌲 Random Forest Classifier

* Trained a **Random Forest** with 100 decision trees
* Achieved better accuracy and stability
* More resistant to overfitting

---

### 5. ⭐ Feature Importance

Extracted and visualized the most important features contributing to predictions:

```python
# Example Output:
1. cp (chest pain type)
2. thalach (max heart rate)
3. ca (number of major vessels)
4. exang (exercise-induced angina)
```

These features played the biggest roles in predicting heart disease.

---

### 6. ✅ Cross-Validation

Used **5-fold cross-validation** to check if the model is reliable and consistent:

```python
from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(model, X, y, cv=5)
```

---

## 📈 Results Summary

| Model         | Test Accuracy | Cross-Val Accuracy | Notes                            |
| ------------- | ------------- | ------------------ | -------------------------------- |
| Decision Tree | \~82%         | \~78%              | Overfitting reduced with pruning |
| Random Forest | \~90%         | \~88%              | More robust and stable           |

---

## 💬 Key Takeaways

* 🌿 **Decision Trees** are easy to understand and visualize — great for explanations.
* 🌲 **Random Forests** combine many trees to create more reliable predictions.
* 🔍 Feature importance helps explain "why" a model makes a decision.
* ✅ Cross-validation checks how well your model generalizes to unseen data.

---

## 🗂 File Structure

```bash
heart-disease-prediction/
├── heart.csv
├── task5_heart_disease_tree_models.py   # Main script
├── tree.dot                             # (Optional Graphviz export)
├── tree.png                             # (Optional tree image)
└── README.md
```

---

## 🚀 Future Enhancements

* Use `GridSearchCV` for tuning hyperparameters like `max_depth`, `n_estimators`, etc.
* Try gradient boosting algorithms like `XGBoost` or `LightGBM`
* Build a small web app using Flask or Streamlit to make it interactive
* Deploy the model using AWS or Heroku
