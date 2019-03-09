import pandas as pd
import matplotlib.pyplot as plt
import DataImport as di
import CommentVectorizer as cv

from sklearn.decomposition import PCA, TruncatedSVD


def load_train_data():
    return di.load_train_data()


def get_basic_properties(toxic_comments_data: pd.DataFrame):
    total_rows = toxic_comments_data.shape[0]
    pi = dict()
    print('Total number of data points: {0}'.format(total_rows))
    print('Classes in the comment data set: {0}'.format(di.CATEGORIES))
    for comment_class in di.CATEGORIES:
        pi[comment_class] = len(toxic_comments_data[toxic_comments_data[comment_class] == 1])/total_rows
    print("Probability of each class:")
    print("\n".join("{0}: {1}".format(k, v) for k, v in pi.items()))


def get_comment_vectors(toxic_comments_data):
    comment_vectorizer = cv.CommentVectorizer()
    vectorizer = comment_vectorizer.get_count_vectorizers()
    comment_vectorizer.fit(toxic_comments_data[di.TEXT_COLUMN], vectorizer)
    return comment_vectorizer.transform(toxic_comments_data[di.TEXT_COLUMN], vectorizer)


def plot_reduced_comment_vectors(comment_vectors):
    two_d_vectors = TruncatedSVD(n_components=2).fit_transform(comment_vectors)
    plt.scatter(two_d_vectors[:, 0], two_d_vectors[:, 1], color='red')
    plt.title("2D Plot of Reduced Comment Vectors (using SVD)")
    plt.xlabel("Reduced Feature 1")
    plt.ylabel("Reduced Feature 2")
    plt.savefig("2D Plot of Reduced Comment Vectors (using SVD).pdf")


# toxic_comments_train = load_train_data()
# get_basic_properties(toxic_comments_train)
#
# comment_vectors = get_comment_vectors(toxic_comments_train)
#
# plot_reduced_comment_vectors(comment_vectors)
