import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('max_columns', 20)


def clean_data(file):
    
    df = pd.read_csv(file)
    
    sns.heatmap(df.isnull(), yticklabels=False)
    plt.show()
    
    passengers = df['PassengerId'].tolist()
    
    df.drop('PassengerId', axis=1, inplace=True)
    
    pclass = pd.get_dummies(df['Pclass'], prefix='Pclass', drop_first=True)
    
    sex = pd.get_dummies(df['Sex'], prefix='Sex', drop_first=True)
    
    avg_age = df.groupby(['Pclass'])['Age'].mean()
    class_list = df['Pclass'].unique().tolist()
    age_map = {c: avg_age[c] for c in class_list}
    
    df['Age'] = df['Age'].fillna(df['Pclass'].map(age_map))
    
    df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
    
    embark = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
    
    parse_names = df['Name'].apply(lambda s: s.split(',')[1].split('.')[0])
    title_dict = parse_names.value_counts().to_dict()
    minimum_title = sorted(list(title_dict.values()), reverse=True)[3]
    
    title_map = {}
    for title in title_dict.keys():
        if title_dict[title] >= minimum_title:
            title_map[title] = title_dict[title]
    
    titles = parse_names.apply(lambda x: x if x in title_map else 'Other')
    title = pd.get_dummies(titles, prefix='Title', drop_first=True)
    
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    
    df.drop(['Pclass', 'Name', 'Sex', 'Embarked'], axis=1, inplace=True)
    
    df = pd.concat([df, pclass, sex, embark, title], axis=1)
    
    df = df.dropna()
    
    if file == 'test.csv':
        return df, passengers  
    else:
        return df
    

def train_model(train_df, model_type, test_exist='Y'):
    
    if test_exist == 'N':
    
        from sklearn.model_selection import train_test_split
        
        X = train_df.drop('Survived', axis=1)
        y = train_df['Survived']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        
    else:

        from sklearn.utils import shuffle
        
        train_df = shuffle(train_df)
        
        X_train = train_df.drop('Survived', axis=1)
        y_train = train_df['Survived']
    
    if model_type == 'decision_tree':
        
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        
    elif model_type == 'random_forest':
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300)
        
    elif model_type == 'logistic_reg':
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        
    model.fit(X_train, y_train)
    
    return model
        

def generate_predictions():

    training_data = clean_data('train.csv')
    
    trained_decisiontree = train_model(training_data, 'decision_tree')
    trained_randomforest = train_model(training_data, 'random_forest')
    trained_logisticreg = train_model(training_data, 'logistic_reg')
    
    testing_data, testing_passengers = clean_data('test.csv')
    
    predictions_decisiontree = trained_decisiontree.predict(testing_data)
    predictions_randomforest = trained_randomforest.predict(testing_data)
    predictions_logisticreg = trained_logisticreg.predict(testing_data)
    
    submission = pd.DataFrame()
    
    submission['PassengerId'] = testing_passengers
    submission['DecisionTree_Survived'] = predictions_decisiontree
    submission['RandomForest_Survived'] = predictions_randomforest
    submission['LogisticReg_Survived'] = predictions_logisticreg
    
    submission.to_csv('submission.csv', index=False)
    
    return submission


def evaluate_model(predictions, actual):
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(classification_report(actual, predictions))
    print('\n')
    print(confusion_matrix(actual, predictions))


generate_predictions()




