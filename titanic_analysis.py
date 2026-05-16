import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "titanic.csv"
JSON_FILE = DATA_DIR / "titanic_data.json"
DATA_DIR.mkdir(exist_ok=True)

print("Project setup complete!")
print(f"Data directory: {DATA_DIR}")
print(f"CSV file location: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)
print(f"Dataset loaded successfully! Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS")
print("="*50)

numeric_columns = df.select_dtypes(include=[np.number]).columns

for col in numeric_columns:
    print(f"\nColumn: {col}")
    print(f"  Mean:   {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Std:    {df[col].std():.2f}")

print("\n" + "="*50)
print("MISSING VALUES ANALYSIS")
print("="*50)

missing_data = {}
for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_data[col] = {
        'missing_count': int(missing_count),
        'missing_percent': round(missing_percent, 2)
    }
    if missing_count > 0:
        print(f"{col}: {missing_count} missing ({missing_percent:.1f}%)")

df_features = df.copy()
df_features['FamilySize'] = df_features['SibSp'] + df_features['Parch'] + 1
print(df_features[['SibSp', 'Parch', 'FamilySize']].head(10))

df_features['IsAlone'] = (df_features['FamilySize'] == 1).astype(int)
print(df_features[['FamilySize', 'IsAlone']].head(10))

def categorize_age(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 18:
        return 'Child'
    elif age < 30:
        return 'Young Adult'
    elif age < 50:
        return 'Adult'
    else:
        return 'Senior'

df_features['AgeGroup'] = df_features['Age'].apply(categorize_age)
print(df_features[['Age', 'AgeGroup']].head(10))

print("\n" + "="*50)
print("FEATURE ANALYSIS: SURVIVED vs NOT SURVIVED")
print("="*50)

print("\nFamily Size by Survival:")
family_survival = df_features.groupby('Survived')['FamilySize'].agg(['mean', 'median', 'std'])
print(family_survival)

print("\n" + "="*50)
print("FEATURE DIFFERENTIATION ANALYSIS")
print("="*50)

survived = df_features[df_features['Survived'] == 1]
not_survived = df_features[df_features['Survived'] == 0]

print("\nFamily Size:")
print(f"  Survived mean: {survived['FamilySize'].mean():.2f}")
print(f"  Not Survived mean: {not_survived['FamilySize'].mean():.2f}")
print(f"  Difference: {abs(survived['FamilySize'].mean() - not_survived['FamilySize'].mean()):.2f}")

print("\nIs Alone:")
print(f"  Survived - alone rate: {survived['IsAlone'].mean():.2%}")
print(f"  Not Survived - alone rate: {not_survived['IsAlone'].mean():.2%}")

df_engineered = df_features.copy()
print("\nFeature engineering complete!")

class Passenger:
    def __init__(self, passenger_id, name, age, sex, survived, pclass,
                 fare, embarked=None, family_size=None, is_alone=None, title=None):
        self.passenger_id = int(passenger_id) if pd.notna(passenger_id) else None
        self.name = str(name) if pd.notna(name) else None
        self.age = float(age) if pd.notna(age) else None
        self.sex = str(sex) if pd.notna(sex) else None
        self.survived = int(survived) if pd.notna(survived) else None
        self.pclass = int(pclass) if pd.notna(pclass) else None
        self.fare = float(fare) if pd.notna(fare) else None
        self.embarked = str(embarked) if pd.notna(embarked) else None
        self.family_size = int(family_size) if pd.notna(family_size) else None
        self.is_alone = int(is_alone) if pd.notna(is_alone) else None
        self.title = str(title) if pd.notna(title) else None

    def to_dict(self):
        return {
            'passenger_id': self.passenger_id,
            'name': self.name,
            'age': self.age,
            'sex': self.sex,
            'survived': self.survived,
            'pclass': self.pclass,
            'fare': self.fare,
            'embarked': self.embarked,
            'family_size': self.family_size,
            'is_alone': self.is_alone,
            'title': self.title
        }


class TitanicDataset:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.passengers = []
        self._create_passengers()

    def _create_passengers(self):
        for idx, row in self.dataframe.iterrows():
            passenger = Passenger(
                passenger_id=row.get('PassengerId', idx),
                name=row.get('Name', 'Unknown'),
                age=row.get('Age', None),
                sex=row.get('Sex', None),
                survived=row.get('Survived', None),
                pclass=row.get('Pclass', None),
                fare=row.get('Fare', None),
                embarked=row.get('Embarked', None),
                family_size=row.get('FamilySize', None),
                is_alone=row.get('IsAlone', None),
                title=row.get('Title', None)
            )
            self.passengers.append(passenger)

    def to_json(self, filename='titanic_data.json'):
        data = {
            'metadata': {
                'dataset_name': 'Titanic Passenger Dataset',
                'export_date': datetime.now().isoformat(),
                'total_passengers': len(self.passengers),
                'survival_rate': round(self.dataframe['Survived'].mean(), 4)
            },
            'passengers': [p.to_dict() for p in self.passengers]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Data exported to {filename}")
        return data

    def get_summary_stats(self):
        survived_count = sum(1 for p in self.passengers if p.survived == 1)
        not_survived_count = sum(1 for p in self.passengers if p.survived == 0)
        ages = [p.age for p in self.passengers if p.age is not None]
        fares = [p.fare for p in self.passengers if p.fare is not None]
        return {
            'total_passengers': len(self.passengers),
            'survived': survived_count,
            'did_not_survive': not_survived_count,
            'average_age': round(sum(ages) / len(ages), 2) if ages else None,
            'average_fare': round(sum(fares) / len(fares), 2) if fares else None
        }


if 'df_engineered' in locals() and not df_engineered.empty:
    dataset = TitanicDataset(df_engineered)
    print(f"Dataset created with {len(dataset.passengers)} passengers")
    stats = dataset.get_summary_stats()
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    dataset.to_json('titanic_data.json')

with open('titanic_data.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

print("\n" + "="*50)
print("JSON VALIDATION")
print("="*50)
print("JSON loaded successfully!")
print(f"Dataset name: {json_data['metadata']['dataset_name']}")
print(f"Export date: {json_data['metadata']['export_date']}")
print(f"Total passengers: {json_data['metadata']['total_passengers']}")
print(f"Survival rate: {json_data['metadata']['survival_rate']:.2%}")
print(f"Number of passenger records: {len(json_data['passengers'])}")
print("\nFirst passenger preview:")
print(json.dumps(json_data['passengers'][0], indent=2))
