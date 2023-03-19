# Project-Bondora-Financial-risk-modelling-of-European-P2P-lending-platform
The main purposes of this analysis are to summarize the characteristics of variables that can affect the loan status and to get some ideas about the relationships among variables.

## Feature Engineering and Data Pre-Processing:
### Pearson Correlation
The Pearson correlation measures the strength of the linear relationship between two variables. It has a value between -1 to 1, 
* -1 meaning a total negative linear correlation 
* 0 being no correlation, and + 1 meaning a total positive correlation
![Markdown](https://editor.analyticsvidhya.com/uploads/39170Formula.JPG)

```Python
def correlation(data, threshold):
  feature_correlation = set()
  correlation_matrix = data.corr()
  for i in range(len(correlation_matrix.columns)):
    for j in range(i):
      if abs(correlation_matrix.iloc[i, j]) > threshold:
        column_name = correlation_matrix.columns[i]
        feature_correlation.add(column_name)
  return feature_correlation
```
The function mentioned above accepts a dataset in the form of a pandas DataFrame, along with a threshold value. It then identifies and returns a set of features that have a correlation coefficient higher than the specified threshold. To avoid overfitting our model, we can choose to remove columns that do not contribute significantly to the model's accuracy.

```Python
current_Index = df[df['Status']=='Current'].index
data = df.drop(current_Index)
data['Status'] = data['Status'].replace({'Late':'Default','Repaid':'Not Default'})
```
We are dropping all the rows with the `Status` column equal to `Current` from our `df` and assigning the remaining data to a new DataFrame named `data`. Then, we are replacing the values `Late` with `Default` and `Repaid` with `Not Default` in the `Status` column of the `data` DataFrame.

```Python
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
```
The function mentioned above calculates mutual information scores using the mutual_info_classif function from scikit-learn, which measures the dependence between two variables. The function returns a pandas Series object called mi_scores containing the mutual information scores for each feature in X, sorted in descending order.

We chose the top 15 features for our analysis. We selected 10 features based on their mi scores and created the remaining 3 target variables from the remaining 5 features.
 * InterestAndPenaltyBalance
 * PrincipalBalance
 * PrincipalPaymentsMade
 * PrincipalOverdueBySchedule
 * NextPaymentNr
 * StageActiveSince_week
 * StageActiveSince_month
 * StageActiveSince_day
 * NrOfScheduledPayments
 * MaturityDate_Last_year
 * Amount
 * Interest
 * IncomeTotal
 * LiabilitiesTotal
 * LoanDuration