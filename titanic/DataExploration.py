import DataImport as di
import matplotlib.pyplot as plt
import seaborn as sns


def plot_survival_by_gender():
    train_data = di.get_titanic_data()
    gender_table = train_data.pivot_table(index="Sex", values="Survived")
    gender_table.plot.bar()
    plt.title("Graph to Show Survival Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Survival Rate")
    plt.savefig("Graph to Show Survival Rate by Gender")


def plot_survival_by_class():
    train_data = di.get_titanic_data()
    pclass_table = train_data.pivot_table(index="Pclass", values="Survived")
    pclass_table.plot.bar()
    plt.title("Graph to Show Survival Rate by Class")
    plt.xlabel("Class")
    plt.ylabel("Survival Rate")
    plt.savefig("Graph to Show Survival Rate by Class")


def plot_survival_by_age():
    train_data = di.get_titanic_data()
    survived = train_data[train_data["Survived"] == 1]
    died = train_data[train_data["Survived"] == 0]
    survived["Age"].plot.hist(alpha=0.5, color="green", bins=50)
    died["Age"].plot.hist(alpha=0.5, color="red", bins=50)
    plt.title("Graph to Show Survival Rate by Age")
    plt.xlabel("Age (years)")
    plt.ylabel("Survival Rate")
    plt.legend(["Survived", "Died"])
    plt.savefig("Graph to Show Survival Rate by Age")


def plot_survival_by_age_category():
    train_data = di.get_titanic_data()
    train_data = di.filter_age(train_data)
    age_categories_pivot = train_data.pivot_table(index="Age_category", values="Survived")
    age_categories_pivot.plot.bar()
    plt.title("Graph to Show Survival Rate by Age Category")
    plt.xlabel("Age Group")
    plt.ylabel("Survival Rate")
    plt.savefig("Graph to Show Survival Rate by Age Category")


def plot_survival_by_fare():
    train_data = di.get_titanic_data()
    plt.figure(figsize=(15, 8))
    died_fare = train_data.loc[(train_data["Survived"] == 0), "Fare"]
    survived_fare = train_data.loc[(train_data["Survived"] == 1), "Fare"]
    sns.kdeplot(died_fare, color="gray", shade=True, label="Not Survived")
    sns.kdeplot(survived_fare, color="g", shade=True, label="Survived")
    plt.title("Graph to Show Survival Rate by Fare")
    plt.xlabel("Fare")
    plt.ylabel("Frequency of Passengers")
    plt.savefig("Graph to Show Survival Rate by Fare")


def plot_feature_heatmap():
    train_data = di.get_titanic_data()
    plt.figure(figsize=(14, 12))
    correlation_matrix = train_data.corr(method='spearman')
    sns.heatmap(correlation_matrix, square=True, annot=True)
    plt.title("Heatmap of Correlation of Different Features", size=24)
    plt.xlabel("Feature", size=20)
    plt.ylabel("Feature", size=20)
    plt.savefig("Heatmap of Correlation of Different Features")


plot_feature_heatmap()
