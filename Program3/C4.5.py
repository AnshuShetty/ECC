import pandas as pd
import graphviz
from C45 import C45Classifier

data = pd.read_csv('id3.csv')
X = data.drop(['PlayTennis'], axis=1)
y = data['PlayTennis']

model = C45Classifier()
model.fit(X, y)

data_test = pd.read_csv('data_test.csv')
model.predict(data_test)


data_test = pd.read_csv('data_test_c.csv')
X_test = data_test.drop(['target'], axis=1)
y_test = data_test['target']
model.evaluate(X_test, y_test)

model.generate_tree_diagram(graphviz,"decision_tree")