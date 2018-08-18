import DataImport as di
import matplotlib.pyplot as plt

train_data = di.get_titanic_data()


def plot_survival_by_gender():
    gender_table = train_data.pivot_table(index="Sex", values="Survived")
    gender_table.plot.bar()
    plt.title("Graph to Show Survival Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Survival Rate")
    plt.savefig("Graph to Show Survival Rate by Gender")


def plot_survival_by_class():
    pclass_table = train_data.pivot_table(index="Pclass", values="Survived")
    pclass_table.plot.bar()
    plt.title("Graph to Show Survival Rate by Class")
    plt.xlabel("Class")
    plt.ylabel("Survival Rate")
    plt.savefig("Graph to Show Survival Rate by Class")


def plot_survival_by_age():
    survived = train_data[train_data["Survived"] == 1]
    died = train_data[train_data["Survived"] == 0]
    survived["Age"].plot.hist(alpha=0.5, color="green", bins=50)
    died["Age"].plot.hist(alpha=0.5, color="red", bins=50)
    plt.title("Graph to Show Survival Rate by Age")
    plt.xlabel("Age (years)")
    plt.ylabel("Survival Rate")
    plt.legend(["Survived", "Died"])
    plt.savefig("Graph to Show Survival Rate by Age")
