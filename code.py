import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

df = pd.read_excel("/content/Unseen_School_Data_20000_Rows.xlsx")
df['Average_Score'] = df[['Math_Score', 'English_Score', 'Science_Score']].mean(axis=1)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df[df['Age'] > 0]

df['Math_Score'].hist(bins=20)
plt.title("Math Score Distribution")
plt.show()

sns.scatterplot(x='Attendance (%)', y='Math_Score', data=df)
plt.title("Attendance vs Math Score")
plt.show()

sns.pairplot(df[['Math_Score', 'English_Score', 'Science_Score', 'Attendance (%)']])
plt.show()

features = ['Math_Score', 'English_Score', 'Science_Score', 'Attendance (%)', 'Age']
X = df[features]
y = df['Average_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Average_Score'], cmap='viridis')
plt.title("PCA Projection")
plt.colorbar()
plt.show()

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print("Lasso Coefficients:", lasso.coef_)

print("Slope (coefficients):", lr.coef_)
print("Intercept:", lr.intercept_)

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_xgb_pred = xgb.predict(X_test)

y_test_binary = (y_test > 75).astype(int)
y_pred_binary = (y_pred > 75).astype(int)
print(confusion_matrix(y_test_binary, y_pred_binary))

sgd = SGDRegressor(max_iter=1000, learning_rate='invscaling')
sgd.fit(X_train, y_train)
print("SGD Coefficients:", sgd.coef_)

scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
print("Cross-Validation R² Scores:", scores)
print("Average R²:", scores.mean())

print(df[features].corr())
print(df[features].cov())
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

grade_avg = df.groupby('Grade')[['Math_Score', 'English_Score', 'Science_Score']].mean()
print(grade_avg)
grade_avg.plot(kind='bar', figsize=(12, 6), title='Average Subject Scores by Grade')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(grade_avg.mean(axis=1).idxmin())
print(grade_avg.var().idxmax())

sns.scatterplot(data=df, x='Attendance (%)', y='Math_Score')
plt.title("Attendance vs Math Score")
plt.show()

filtered = df[(df['Attendance (%)'] < 70) & (df['Average_Score'] > 80)]
print(filtered[['Student_ID', 'Average_Score', 'Attendance (%)']])

df.groupby('Grade')['Attendance (%)'].mean().plot(kind='line', marker='o', title='Average Attendance by Grade')
plt.ylabel("Attendance (%)")
plt.show()

gender_avg = df.groupby('Gender')[['Math_Score', 'English_Score', 'Science_Score']].mean()
print(gender_avg)

sns.boxplot(data=df, x='Gender', y='Math_Score')
plt.title('Math Score Distribution by Gender')
plt.show()

print(df.groupby('Gender')['Science_Score'].mean().idxmax())
print(df.groupby('Gender')['Average_Score'].mean())

state_avg = df.groupby('State')[['Math_Score', 'English_Score', 'Science_Score']].mean()
print(state_avg.head())

state_avg.plot(kind='bar', figsize=(14, 6), title='Average Scores by State')
plt.ylabel("Scores")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

avg_scores_by_state = state_avg.mean(axis=1)
print(avg_scores_by_state.idxmax())
print(avg_scores_by_state.idxmin())

state_attendance = df.groupby('State')[['Attendance (%)', 'Math_Score']].mean()
sns.scatterplot(data=state_attendance, x='Attendance (%)', y='Math_Score')
plt.title('State-wise Attendance vs Math Score')
plt.show()

df.groupby('Grade')['Average_Score'].mean().plot(kind='line', marker='o', title="Average Score by Grade")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.show()

sns.regplot(data=df, x='Age', y='Average_Score')
plt.title("Age vs Average Score")
plt.show()

df.groupby('Grade')['Attendance (%)'].mean().plot(kind='bar', title="Attendance by Grade")
plt.ylabel("Attendance (%)")
plt.xticks(rotation=45)
plt.show()

print(df.groupby('Grade')['Age'].agg(['min', 'max']))

df[['Math_Score', 'English_Score', 'Science_Score']].hist(figsize=(12, 6), bins=20)
plt.suptitle("Subject-wise Score Distribution")
plt.show()

subject_std = df[['Math_Score', 'English_Score', 'Science_Score']].std()
print(subject_std.idxmin())

sns.boxplot(data=df[['Math_Score', 'English_Score', 'Science_Score']])
plt.title("Score Comparison Across Subjects")
plt.show()

high_scores = (df[['Math_Score', 'English_Score', 'Science_Score']] > 90).sum()
print(high_scores.idxmax())
