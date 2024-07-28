import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
import graphviz

# Load the dataset

data = pd.read_csv('id3.csv')
df = pd.DataFrame(data)

# Convert categorical variables to numeric using LabelEncoder
label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# Separate features and target variable
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')

# Train the classifier
clf.fit(X, y)

# Visualize the decision tree
dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=label_encoder.classes_,
                           filled=True, rounded=True, special_characters=True, node_ids=True)
graph = graphviz.Source(dot_data)
graph.view()