# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv('tested.csv')

# Basic Information
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Value Counts
print(df['Survived'].value_counts())
print(df['Pclass'].value_counts())
print(df['Sex'].value_counts())

# Univariate Analysis
df.hist(figsize=(12,10))
plt.suptitle('Histograms of Titanic Features')
plt.show()

# Boxplot for Age
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.show()

# Bivariate Analysis
sns.pairplot(df, hue='Survived')
plt.suptitle('Pairplot showing relationships', y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Specific Insights

# Survival by Gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

# Survival by Pclass
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Count by Passenger Class')
plt.show()

# Age distribution by Survival
sns.histplot(data=df, x='Age', hue='Survived', kde=True)
plt.title('Age Distribution by Survival')
plt.show()

# Observations (write in comments)
# - Females had higher survival rates than males.
# - First-class passengers were more likely to survive.
# - Younger passengers (children) had better survival rates.
# - Fare shows significant outliers.
# - Cabin has many missing values.
