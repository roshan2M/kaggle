import DataImport as di
import matplotlib.pyplot as plt


def plot_survival_by_gender():
    train_data = di.get_titanic_data()
    gender_table = train_data.pivot_table(index='Sex', values='Survived')
    gender_table.plot.bar()
    plt.savefig("Graph to Show Survival Rate by Gender")


def plot_survival_by_class():
    train_data = di.get_titanic_data()
    pclass_table = train_data.pivot_table(index='Pclass', values='Survived')
    pclass_table.plot.bar()
    plt.savefig("Graph to Show Survival Rate by Class")
