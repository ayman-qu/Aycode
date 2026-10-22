
Title : "CAR PRICE PREDICTIVE MODEL"

Project by : ayman-qu

Short Intro : This is First ever Data Science Project that i worked on with learning the Basic Machine Learning techniques and different Supervised Algorithms.

Credits for this project : @onurdatascience (youtube tutorial)

Whats this About?

The project simply covers the data science rules, machine learning algorithms , and demonstrating the accuracies of some of the famous models 
in machine learning to know the best results that can be achieved for the Cars predictions. If this sounds Interesting, the best part comes when you are about to start with 
Data Visualisations; like Storytelling but with Visuals on Screen, ( Example: Charts, Graphs, Bars etc..).

You are a Beginner?  ---> No Worries, Skip to the end!

Code Snippets!

This is a Heatmap 
``` python
df = pd.read_csv("C:\\Users\\Dell\\Downloads\\Dataset files\\car_price_dataset.csv")

numeric_col = df.select_dtypes(include = ["Int64", "float64"])
correlation_matrix = numeric_col.corr() # corr() works for only numeric columns

# HeatMap 
sns.heatmap(correlation_matrix, cmap = "coolwarm", linewidth = .5, annot = True)
plt.title("Correlation Heatmap")
plt.show()
```

<img width="1280" height="612" alt="Heatmap" src="https://github.com/user-attachments/assets/1eac324c-f903-40e5-b8fe-28728d82295d" />

Random Forest Regressor
```python

from sklearn.ensemble import RandomForestRegressor
rfrmodel = RandomForestRegressor(oob_score=True)
rfrmodel.fit(scaled_X_train, Y_train)
print("Score =", rfrmodel.oob_score_)

rfrpredicts = rfrmodel.predict(scaled_X_test)

modelresults(rfrpredicts)
```

```python
Score = 0.930968129385923
Mean Absolute Error on model is 704.5697700000001
Root Mean Squared Error on model is 821.464801505944
```

For Beginners

Note: Maybe this would be something you can understand and do it yourself, OR Maybe you are not at all able to comprehend this.
Either way, Never Let yourself Down, If you are able to Code, Get curious When stuck, Then you are a Programmer! Unless you Start doing it Yourself, you are never behind!
