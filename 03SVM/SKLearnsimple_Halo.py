from sklearn import svm

x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]

clf = svm.SVC(kernel='linear')
clf.fit(x, y)

print(clf)


# get support vector
print(clf.support_vectors_)

# get indices support vector
print(clf.support_)

# get number of support vector for each class
print(clf.n_support_)

