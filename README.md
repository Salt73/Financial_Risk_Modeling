# Project-Bondora-Financial-risk-modelling-of-European-P2P-lending-platform
The main purposes of this analysis are to summarize the characteristics of variables that can affect the loan status and to get some ideas about the relationships among variables.

## Project Summary
### Abstract
In this project we will be doing credit risk modelling of peer to peer lending Bondora systems.Data for the study has been taken from a leading European P2P lending platform [Bondora](https://www.bondora.com/en/public-reports#dataset-file-format). The retrieved data is a pool of both defaulted and non-defaulted loans from the time period between 1st March 2009 and 27th January 2020. The data comprises of demographic and financial information of borrowers, and loan transactions.In P2P lending, loans are typically uncollateralized and lenders seek higher returns as a compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision, and realize the return that compensates for the risk.

### Background of Understanding the Problem
Peer-to-peer lending has attracted considerable attention in recent years, largely because it offers a novel way of connecting borrowers and lenders. But as with other innovative approaches to doing business, there is more to it than that. Some might wonder, for example, what makes peer-to-peer lending so different–or, perhaps, so much better–than working with a bank, or why has it become popular in many parts of the world.

Certainly, the industry has witnessed strong growth in recent years. According to Business Insider, transaction volumes in the U.S. and Europe, the world’s leading P2P markets, have expanded at double and, in some cases, triple-digit percentage rates, bolstered by widespread acceptance of doing business online and a supportive regulatory environment.

For investors, "peer-2-peer lending," or "P2P," offers an attractive way to diversify portfolios and enhance long-term performance. When they invest through a peer-to-peer platform, they can profit from an asset class that has proven itself in both good times and bad. Equally important, they can avoid the risks associated with putting all their eggs in one basket, especially at a time when many experts believe that traditional favorites such as stocks and bonds are riskier than ever.

Default risk has long been a significant risk factor to test borrowers’ behaviour in Peer-to-Peer (P2P) lending. In P2P lending, loans are typically uncollateralized and lenders seek higher returns as compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision and realize the return that compensates for the risk.

As in the financial research domain, there are very few datasets available that can be utilized for building and analyzing credit risk models. This dataset will help the research community in building and performing research in the credit risk domain.

[Reference](https://technocollabs.gitbook.io/bondora-statistics/)


## Feature Engineering and Data Pre-Processing:
### Pearson Correlation
The Pearson correlation measures the strength of the linear relationship between two variables. It has a value between -1 to 1, 
* -1 meaning a total negative linear correlation 
* 0 being no correlation, and + 1 meaning a total positive correlation
![Markdown](https://editor.analyticsvidhya.com/uploads/39170Formula.JPG)

**Reasons why a loan could be rejected:**
* Credit score was too low
* Debt-to-income ratio was too high
* Tried to borrow too much
* Income was insufficient or unstable
* Didn’t meet the basic requirements
* Missing information on the application
* Loan purpose didn’t meet the lender’s criteria

[Reference](https://www.lendingtree.com/personal/reasons-why-your-personal-loan-was-declined/)

## Data Wrangling

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

### Mutual Information
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
